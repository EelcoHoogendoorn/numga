"""Drawing for thin-walled bending and twisting: the sections' walls with their centroids and shear
centres; each section with its stress at the clamp, its principal axes, its neutral axis, the load and
the way the tip moves; the bent cantilevers beside the straight beam; a load turning or moving across
the tip; and each section pushed along the beam, with the polars of its nodes and its kern.

Sections are seen from beyond the tip, looking back at the clamp: across to the right and up, read by the planes x
and y. Walls are drawn as thick strokes along their midlines, the axes, the load and the kern as strokes, marks and
fills over them. The beams are drawn on the GPU as each wall swept along the beam, lit strips seen along parallel
rays, the straight beam a faint ghost beside them, all turned from the beam's frame into the drawing's by one rotor.
Coordinates are read out here alone, to frame the section views: points by their pairings with the planes across
and up over their weight."""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, to_rgb

from numga import concatenate, stack
from examples.util.animation import capture
from examples.mechanics.bending import core
from examples.util.wgsl import contours, fitted, image_figure, meshes, show, simplices, solids
from examples.util.wgsl.camera import Camera, flat_panel

mv = core.mv
# Colours of the beam, of the straight beam, of the load, of the neutral axis, of the principal axes,
# and of the kern, and the stress's colour map. The width of a drawn wall and the pieces
# each wall's stress is drawn in. The half width of the section views, the drawn length of the arrows
# in a section and at the tip, and the pieces between drawn cross-sections.
BEAM, STRAIGHT, LOAD, NEUTRAL, PRINCIPAL, KERN = "#0369a1", "#cbd5e1", "#f97316", "#16a34a", "#64748b", "#a855f7"
STRESS = "RdBu_r"
WALL_WIDTH = 5.0
WALL_PIECES = 12
SECTION_LIMIT = 1.2
ARROW, TIP_ARROW = 0.45, 2.4
OUTLINE_EVERY = 10
# The corners of a square, across and up: the clamp's around the walls at the clamp, and, as large as the section
# views, the one the kern is cut from.
SQUARE = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
# The colours of the line of shear centres and of the clamp.
SHEAR_LINE, CLAMP_COLOUR = "#0f172a", "#475569"
# The view of the beams, elevation and azimuth in radians, azimuth zero looking along +y; the margin around them
# as a share of the larger side, the pixels across a view before it is fitted to its axes, how much each side of a straight
# beam's wall hides, and the strips' sheen.
ELEVATION, AZIMUTH = np.radians([22.0, -58.0 + 90.0])
MARGIN = 0.1
PIXELS = 520
STRAIGHT_OPACITY = 0.19
GLOSS = 0.1
# In the section views, in points, at the figure's pixels to the inch: the widths of the principal axes, with their
# dashes and gaps, of the neutral axis, of the load's arrow, with the length of the strokes of the chevron at its
# tip, and of the kern's outline; the sides of the centroid's dot, of the shear centre's ring, with its rim's width,
# and of the push point's cross, with its arms' width and their white rim; and how much the kern's fill covers.
PRINCIPAL_WIDTH, PRINCIPAL_DASHES = 1.0, (3.7, 1.6)
NEUTRAL_WIDTH, LOAD_WIDTH, LOAD_CHEVRON, KERN_WIDTH = 1.8, 2.0, 8.0, 1.4
DOT_SIDE, RING_SIDE, RING_WIDTH, PUSH_SIDE, PUSH_WIDTH, PUSH_RIM = 3.0, 7.0, 1.4, 9.0, 2.4, 1.0
KERN_OPACITY = 0.3
# The radii of the shear centres' dashes and of the clamp's rods, and the width of the load's arrow with the length
# of the strokes of the chevron at its tip, in pixels.
DASH_RADIUS, CLAMP_RADIUS = 0.85, 1.2
SHAFT_WIDTH, SHAFT_CHEVRON = 3.4, 13.6
# The turn carrying the beam's frame, along z with the section across x and up y, into the drawing's, along x with
# the section across y and up z: a quarter turn of z towards x, then of y towards z.
drawing_turn = (mv.yz * (-np.pi / 4)).exp() * (mv.zx * (-np.pi / 4)).exp()   # [] Motor+
# The intrinsic scalar one: a meet of planes, extrinsic, times it is the intrinsic point it names.
unit_volume = mv.antipseudoscalar([1.0])                                     # [] Antipseudoscalar-


# --- plumbing -------------------------------------------------------------------------
def flat(points: core.Point) -> np.ndarray:
    """Points in the section views `[..., 2]`: across and up, read by their pairings with the planes x and y over
    their weight, which do not see any part along the beam."""
    points = points.batch()
    return ((stack((mv.x, mv.y)) & points[..., None]) / (mv.w & points)[..., None]).to_array()


def reach(start: core.Point, direction: core.Point, length: float) -> core.Point:
    """The point a length from a start of unit weight along a direction: the direction made as long as the length, a
    direction's length being the size of the plane square to it."""
    return start + direction / direction.dual().norm() * length


def line_ends(line: core.Strain, centre: core.Point, half: float) -> core.Point:
    """The ends `[2]` of a line of the section drawn for `half` either side of the foot of `centre` on it: the foot
    moved both ways along the line's direction, where it meets the plane at infinity; normalized, the foot keeps the
    sign of its weight, which at most swaps the two ends."""
    foot = ((centre | line) ^ line).normalized()                             # [] Point-
    direction = (line ^ mv.w) * unit_volume                                  # [] Point-
    return reach(foot, direction, half * np.array([-1.0, 1.0]))


def wall_stress(stress: core.Scalar) -> np.ndarray:
    """The stress along the beam at the middle of each piece of each wall `[walls * pieces]`, linear along each wall
    between its ends, from the stress at the ends `[ends] Scalar+[walls]`."""
    tails, heads = stress.to_array()                                         # [walls], [walls]
    fractions = np.linspace(0, 1, WALL_PIECES + 1)
    middles = (fractions[:-1] + fractions[1:]) / 2
    return (tails[:, None] * (1 - middles) + heads[:, None] * middles).reshape(-1)


def walls_drawn(walls: core.Point, colours: np.ndarray, scale: float, camera: Camera) -> np.ndarray:
    """Walls, their ends `[walls, ends]`, as thick strokes, each in its colour `[walls, 3]`."""
    return simplices.strokes(walls.reshape(-1), simplices.Mesh.chains([2] * walls.shape[0]), colours,
                             WALL_WIDTH * scale, camera)


def stress_drawn(walls: core.Point, stress: core.Scalar, largest: float, scale: float, camera: Camera) -> np.ndarray:
    """The walls, their ends `[walls, ends]`, as thick strokes coloured by the stress along the beam, linear along each
    wall between its ends."""
    fractions = np.linspace(0, 1, WALL_PIECES + 1)                           # [pieces + 1]
    along = walls[:, None, 0] * (1 - fractions) + walls[:, None, 1] * fractions   # [walls, pieces + 1] Point-
    colours = plt.get_cmap(STRESS)(Normalize(-largest, largest)(wall_stress(stress)))[:, :3]
    return simplices.strokes(along.reshape(-1), simplices.Mesh.chains([WALL_PIECES + 1] * walls.shape[0]), colours,
                             WALL_WIDTH * scale, camera)


def line_drawn(line: core.Strain, centre: core.Point, half: float, colour: str, width: float, camera: Camera,
               dashes: tuple[float, float] = (1.0, 0.0)) -> np.ndarray:
    """A line of the section, drawn for `half` either side of the foot of `centre` on it, `width` pixels wide."""
    return simplices.strokes(line_ends(line, centre, half), simplices.Mesh.chains([2]), np.array(to_rgb(colour)), width,
                             camera, dashes=dashes, cap=simplices.Cap.BUTT)


def section_panel(ax, limit: float) -> tuple[Camera, float]:
    """A section view's axes framed `limit` either way of the middle, and their camera, reading points by the planes
    across and up; and its pixels to a point."""
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal")
    ax.set_axis_off()
    return flat_panel(ax, core.ga), ax.figure.dpi / 72


def centres_drawn(centre: core.Point, shear: core.Point, scale: float, camera: Camera) -> list[np.ndarray]:
    """The centroid as a dot, and the shear centre as a ring, the centroid's dot last."""
    ink = np.array(to_rgb("#0f172a"))
    # The ring a white dot over a larger dark one, under the centroid's dot, which it rings where the two coincide.
    return [simplices.markers(shear[None], ink, (RING_SIDE + RING_WIDTH) / 2 * scale, camera),
            simplices.markers(shear[None], np.ones(3), (RING_SIDE - RING_WIDTH) / 2 * scale, camera),
            simplices.markers(centre[None], ink, DOT_SIDE / 2 * scale, camera)]


def section_row(sections: list[tuple[core.Mesh, core.Point, core.Scalar]]) -> tuple[plt.Figure, list, list[Camera],
                                                                                   float]:
    """A row of section views, one a section: the figure, its axes, their cameras and its pixels to a point."""
    figure, axes = plt.subplots(1, len(sections), figsize=(4 * len(sections), 3.8), dpi=50)
    for ax in axes:
        ax.set_axis_off()
    figure.tight_layout()
    panels = [section_panel(ax, SECTION_LIMIT) for ax in axes]
    return figure, axes, [camera for camera, _ in panels], panels[0][1]


def draw_sections(sections: list[tuple[core.Mesh, core.Point, core.Scalar]]) -> plt.Figure:
    """The sections' walls along their midlines."""
    figure, axes, cameras, scale = section_row(sections)
    for ax, camera, (mesh, nodes, _) in zip(axes, cameras, sections):
        simplices.overlay(ax, [simplices.strokes(nodes.batch(), mesh, np.array(to_rgb(STRAIGHT)), WALL_WIDTH * scale,
                                                 camera)])
    return figure


def draw_walls(sections: list[tuple[core.Mesh, core.Point, core.Scalar]], centres: list[core.Point],
               shears: list[core.Point]) -> plt.Figure:
    """The sections' walls along their midlines, each with its centroid and its shear centre."""
    figure, axes, cameras, scale = section_row(sections)
    for ax, camera, (mesh, nodes, _), centre, shear in zip(axes, cameras, sections, centres, shears):
        simplices.overlay(ax, [simplices.strokes(nodes.batch(), mesh, np.array(to_rgb(STRAIGHT)), WALL_WIDTH * scale,
                                                 camera), *centres_drawn(centre, shear, scale, camera)])
    return figure


def section_limit(tips: core.Point) -> float:
    """Half the width of section views that hold the clamp's section and the ends of every moved tip's walls
    `[..., walls, ends]` with a load's arrow beside them: their farthest reading across or up, by the planes x and y
    over their weight."""
    ends = tips.batch().reshape(-1)                                          # [ends] Point-
    readings = (stack((mv.x, mv.y))[:, None] & ends) / (mv.w & ends)          # [2, ends] Scalar+
    return max(SECTION_LIMIT, float(readings.abs().max().to_array()) + ARROW)


def section_drawn(walls: core.Point, stress: core.Scalar, principal: core.Strain, centre: core.Point,
                  shear: core.Point, through: core.Point, load: core.Point, tip: core.Point, strain: core.Twist,
                  largest: float, limit: float, scale: float, camera: Camera) -> list[np.ndarray]:
    """One section: the stress along the beam over its walls at the clamp, their ends `[walls, ends]`, the walls at
    the tip as the tip has moved, `tip` alike, its principal axes dashed through the centroid, the neutral axis, and
    the load's direction where it acts on the moved tip, an arrow."""
    pictures = [stress_drawn(walls, stress, largest, scale, camera),
                walls_drawn(tip, np.array(to_rgb(BEAM)), scale, camera)]
    pictures += [line_drawn(axis, centre, 2 * limit, PRINCIPAL, PRINCIPAL_WIDTH * scale, camera,
                            tuple(np.array(PRINCIPAL_DASHES) * PRINCIPAL_WIDTH * scale)) for axis in principal]
    # The neutral axis: the strain's part in the section's plane, the part that stretches the fibres.
    pictures.append(line_drawn(strain.cast(core.Strain), centre, 2 * limit, NEUTRAL, NEUTRAL_WIDTH * scale, camera))
    # The load's arrow, a stroke ending in a chevron.
    tip = reach(through, load, ARROW)                                        # [] Point-
    pictures.append(simplices.strokes(stack([through, tip]), simplices.Mesh.chains([2]), np.array(to_rgb(LOAD)),
                                      LOAD_WIDTH * scale, camera, chevrons=np.ones(1), chevron=LOAD_CHEVRON * scale))
    return pictures + centres_drawn(centre, shear, scale, camera)


def straightened(bent: core.Point) -> core.Point:
    """The straight beam beside bent ones, the ends of their walls at every cross-section `[..., pieces + 1, walls,
    ends]`: the walls at the clamp carried straight along the beam, as far along as each cross-section's middle lies
    along the bent beam."""
    middles = bent.mean(axis=-1).mean(axis=-1)                               # [..., pieces + 1] Point-
    # The distance between neighbouring middles, the size of the line through them, and how far along the beam each
    # middle lies, the distances summed from the clamp.
    steps = (middles[..., :-1] & middles[..., 1:]).norm()                    # [..., pieces] Scalar+
    arc = concatenate([0 * steps[..., :1], steps.cumsum(axis=-1)], axis=-1)  # [..., pieces + 1] Scalar+
    return bent[..., :1, :, :] + core.beam_direction * arc[..., None, None]


def view(bent: list[core.Point], tips: list[core.Point], elevation: float, azimuth: float) -> solids.View:
    """A view from an elevation and azimuth framing every bent beam `[..., pieces + 1, walls, ends]` with its straight
    beam, the clamp, and the loads' tips."""
    points = [core.point(SQUARE)] + bent + [straightened(beam) for beam in bent] + tips
    return solids.view([drawing_turn >> batch for batch in points], 0.0, elevation, azimuth, MARGIN, PIXELS)


def swept(bent: core.Point, colour: str, view: solids.View) -> meshes.Samples:
    """Each wall of a bent beam `[pieces + 1, walls, ends]` swept along it as lit strips, as `meshes.samples`."""
    # Each wall between two cross-sections a quad, two triangles, by the indices of its corners.
    index = np.arange(np.prod(bent.shape)).reshape(bent.shape)               # [pieces + 1, walls, ends]
    quads = np.stack([index[:-1, :, 0], index[:-1, :, 1], index[1:, :, 1], index[1:, :, 0]], axis=-1)
    triangles = quads[..., [[0, 1, 2], [0, 2, 3]]].reshape(-1, 3)             # [pieces * walls * 2, 3]
    places = drawing_turn >> bent.reshape(-1)                                # [vertices] Point-
    return meshes.samples(places, triangles, np.array(to_rgb(colour)), meshes.flat(places, triangles), view,
                          face_values=np.zeros(4), colour=meshes.plain, gloss=GLOSS)


def ghost(bent: core.Point, view: solids.View) -> solids.Plane:
    """The straight beam beside a bent one `[pieces + 1, walls, ends]`: each wall a plate a pixel thick, from its
    ends at the clamp to its ends at the tip, bounded by planes `[walls, 6]`."""
    straight = drawing_turn >> straightened(bent)                            # [pieces + 1, walls, ends] Point-
    return solids.plates(stack([straight[0, :, 0], straight[0, :, 1], straight[-1, :, 1], straight[-1, :, 0]],
                               axis=-1), view.pixel)


def clamped(through: core.Point, load: core.Point, length: float, clamp_radius: float, shaft_width: float,
            view: solids.View) -> tuple[list[tuple[solids.Quadric, np.ndarray]], meshes.Samples]:
    """The clamp, its rods `clamp_radius` pixels in radius, and the tip load where it acts, drawn a length long, a
    tube `shaft_width` pixels across for `solids.draw` to compose among the solids."""
    clamp = drawing_turn >> core.point(SQUARE)                                # [corners] Point-
    clamp_rods, clamp_joints = solids.edges(clamp, clamp[[1, 2, 3, 0]], clamp_radius * view.pixel)
    # The load's arrow, a tube ending in a chevron.
    tip = drawing_turn >> reach(through, load, length)                       # [] Point-
    arrow = simplices.tubes(stack([drawing_turn >> through, tip]), simplices.Mesh.chains([2]), np.array(to_rgb(LOAD)),
                            shaft_width, view, gloss=solids.gloss, chevrons=np.ones(1), chevron=SHAFT_CHEVRON)
    clamp_colour = np.array(to_rgb(CLAMP_COLOUR))
    return [(clamp_rods, clamp_colour), (clamp_joints, clamp_colour)], arrow


def draw_beam(bent: core.Point, shears: core.Point, through: core.Point, load: core.Point,
              view: solids.View) -> np.ndarray:
    """One cantilever, the ends of its walls at every cross-section `[pieces + 1, walls, ends]`: each wall swept
    along the beam as lit strips; the straight beam a faint ghost, each wall a translucent plate; the line of shear
    centres the beam twists about, dashed; the clamp; and the tip load where it acts."""
    # The line of shear centres in dashes, every other piece of it.
    line = drawing_turn >> shears                                            # [pieces + 1] Point-
    dashes = solids.rods(line[:-1:2], line[1::2], DASH_RADIUS * view.pixel)  # [dashes, 3] Plane+ <- Point-
    clamp, arrow = clamped(through, load, TIP_ARROW, CLAMP_RADIUS, SHAFT_WIDTH, view)
    opaque = [(dashes, np.array(to_rgb(SHEAR_LINE)))] + clamp
    return solids.draw(opaque, view, np.ones(3), inks=[],
                       translucent=[(ghost(bent, view), np.array(to_rgb(STRAIGHT)), STRAIGHT_OPACITY)],
                       meshes=[(swept(bent, BEAM, view), 1.0), (arrow, 1.0)])


def draw_beams(sections: list[tuple[core.Mesh, core.Point, core.Scalar]], centres: list[core.Point],
               shears: list[core.Point], load: core.Point,
               beams: list[tuple[core.Twist, core.Motor, core.Strain, core.Scalar]]) -> plt.Figure:
    """Top, each section with its stress at the clamp; below, each cantilever bent under the load
    through its centroid."""
    figure = plt.figure(figsize=(4.4 * len(sections), 8.6), dpi=55)
    count = len(sections)
    section_axes = [figure.add_subplot(2, count, case + 1) for case in range(count)]
    beam_axes = [figure.add_subplot(2, count, count + case + 1) for case in range(count)]
    figure.subplots_adjust(left=0.01, right=0.99, bottom=0.0, top=0.97, hspace=0.0, wspace=0.05)
    largest = max(np.abs(stress.to_array()).max() for _, _, _, stress in beams)
    # Each beam's nodes at every cross-section, the ends of its walls there, and those at the tip.
    placed = [frames[..., None] >> nodes.batch() for (_, nodes, _), (_, frames, _, _) in zip(sections, beams)]
    bent = [moved[..., mesh.edges] for (mesh, _, _), moved in zip(sections, placed)]   # [pieces + 1, walls, ends] Point-
    tips = [reach(frames[-1] >> centre, load, TIP_ARROW) for centre, (_, frames, _, _) in zip(centres, beams)]
    # The beams' rooms alike, one view fitted to the first holds them all.
    seen = fitted(view(bent, tips, ELEVATION, AZIMUTH), beam_axes[0])
    limit = max(section_limit(beam[-1]) for beam in bent)
    for section_ax, beam_ax, (mesh, nodes, _), centre, shear, (strains, frames, principal, stress), beam in zip(
            section_axes, beam_axes, sections, centres, shears, beams, bent):
        camera, scale = section_panel(section_ax, limit)
        simplices.overlay(section_ax, section_drawn(nodes.batch()[mesh.edges], stress, principal, centre, shear,
                                                    frames[-1] >> centre, load, beam[-1], strains[0], largest, limit,
                                                    scale, camera))
        show(beam_ax, draw_beam(beam, frames >> shear, frames[-1] >> centre, load, seen), seen)
    return figure


def animate_loads(section: tuple[core.Mesh, core.Point, core.Scalar], centre: core.Point, shear: core.Point,
                  through: core.Point, loads: core.Point,
                  beam: tuple[core.Twist, core.Motor, core.Strain, core.Scalar]) -> Iterator[np.ndarray]:
    """Each frame of a changing tip load, as `sized_loads` draws it at 50 pixels to the inch."""
    return sized_loads(section, centre, shear, through, loads, beam, 50)


def sized_loads(section: tuple[core.Mesh, core.Point, core.Scalar], centre: core.Point, shear: core.Point,
                through: core.Point, loads: core.Point, beam: tuple[core.Twist, core.Motor, core.Strain, core.Scalar],
                dpi: int) -> Iterator[np.ndarray]:
    """Each frame of a changing tip load, through points `through` in directions `loads`, either batched
    over the frames, at `dpi` pixels to the inch: the bent beam, and its section's stress at the clamp with the tip's
    cross-section."""
    mesh, nodes, _ = section
    strains, frames, principal, stress = beam
    count = frames.shape[0]
    through, loads = through.broadcast_to((count,)), loads.broadcast_to((count,))
    placed = frames[..., None] >> nodes.batch()                              # [frames, pieces + 1, nodes] Point-
    # The ends of the walls at every cross-section.
    bent = placed[..., mesh.edges]                                           # [frames, pieces + 1, walls, ends] Point-
    largest = np.abs(stress.to_array()).max()
    limit = section_limit(bent[:, -1])
    moved = frames[:, -1] >> through                                         # [frames] Point-
    # The figure and the section's still parts drawn once; the beam's picture, the stress, the tip's walls, the
    # neutral axis and the load's arrow set for each frame.
    figure = plt.figure(figsize=(9.0, 4.6), dpi=dpi)
    beam_ax, section_ax = figure.add_subplot(1, 2, 1), figure.add_subplot(1, 2, 2)
    figure.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.98, wspace=0.05)
    # One camera framing every frame's beam.
    seen = fitted(view([bent], [reach(moved, loads, TIP_ARROW)], ELEVATION, AZIMUTH), beam_ax)
    image = show(beam_ax, np.ones((*seen.shape, 3)), seen)
    camera, scale = section_panel(section_ax, limit)
    section = simplices.overlay(section_ax, [np.zeros(camera.shape + (4,))])
    for frame in range(count):
        image.set_data(draw_beam(bent[frame], frames[frame] >> shear, moved[frame], loads[frame], seen))
        section.set_data(simplices.composited(section_drawn(
            nodes.batch()[mesh.edges], stress[frame], principal, centre, shear, moved[frame], loads[frame],
            bent[frame, -1], strains[frame, 0], largest, limit, scale, camera)))
        yield capture(figure)
    plt.close(figure)


def kern(polars: core.Strain, centre: core.Point) -> core.Point:
    """The kern as a polygon `[corners]`: the view cut down to the centroid's side of every node's polar. Each
    polar is read by the stretch it gives the fibre through a point, affine across the section, positive on the
    centroid's side."""
    corners = core.point(SECTION_LIMIT * SQUARE)                             # [corners] Point-
    # Each polar's stretch of the centroid's fibre gives the side that is kept.
    stretch = polars & (core.beam_direction & centre)
    keeping = np.asarray(stretch > 0, dtype=int) - np.asarray(stretch < 0, dtype=int)
    for polar, sense in zip(polars, keeping):
        sides = sense * (polar & (core.beam_direction & corners))   # [corners]
        following = np.roll(np.arange(len(sides)), -1)
        kept = []
        for corner, here, there in zip(range(len(sides)), sides, sides[following]):
            if here >= 0:
                kept.append(corners[corner])
            # Where the polar crosses an edge: the point of the edge its stretch vanishes at, the corners weighted
            # by each other's stretch, at unit weight.
            if here * there < 0:
                kept.append((corners[following[corner]] * here - corners[corner] * there) / (here - there))
        corners = stack(kept)
    return corners


def draw_push(ax, walls: core.Point, point: core.Point, strain: core.Strain, stress: core.Scalar,
              polars: core.Strain, centre: core.Point, largest: float) -> None:
    """One section pushed through one point: the stress over its walls, their ends `[walls, ends]`, the kern, and the
    neutral axis, the push point's polar."""
    camera, scale = section_panel(ax, SECTION_LIMIT)
    corners = kern(polars, centre)                                           # [corners] Point-
    n_corners = len(corners)
    # The kern is convex: a fan of triangles from its first corner fills it.
    fan = simplices.Mesh(n_corners, np.stack([np.zeros(n_corners - 2, dtype=int), np.arange(1, n_corners - 1),
                                              np.arange(2, n_corners)], axis=-1))
    loop = simplices.Mesh(n_corners, np.stack([np.arange(n_corners), np.roll(np.arange(n_corners), -1)], axis=-1))
    kern_colour = np.array(to_rgb(KERN))
    # The push point a cross: two strokes through it, turned an eighth of a turn either way from across, over white
    # strokes a rim wider.
    turns = (mv.xy * (np.pi / 8 * np.array([1.0, -1.0]))).exp()              # [arms] Motor+
    arms = (turns >> mv.x.dual()) * (PUSH_SIDE / 2 * scale * camera.pixel)   # [arms] Point-
    cross = concatenate([point - arms, point + arms])                        # [2 * arms] Point-
    strokes = simplices.Mesh(4, np.array([[0, 2], [1, 3]]))
    simplices.overlay(ax, [
        stress_drawn(walls, stress, largest, scale, camera),
        contours.picture(corners, fan, np.zeros(n_corners), kern_colour, camera, opacity=KERN_OPACITY),
        simplices.strokes(corners, loop, kern_colour, KERN_WIDTH * scale, camera),
        line_drawn(strain, centre, 4 * SECTION_LIMIT, NEUTRAL, NEUTRAL_WIDTH * scale, camera),
        simplices.strokes(cross, strokes, np.ones(3), (PUSH_WIDTH + 2 * PUSH_RIM) * scale, camera),
        simplices.strokes(cross, strokes, np.array(to_rgb(LOAD)), PUSH_WIDTH * scale, camera),
        centres_drawn(centre, centre, scale, camera)[-1]])


def draw_pushes(sections: list[tuple[core.Mesh, core.Point, core.Scalar]], centres: list[core.Point],
                points: core.Point, pushes: list[tuple[core.Strain, core.Scalar, core.Strain]]) -> plt.Figure:
    """A row per push point and a column per section: the section pushed along the beam through the
    point, with its kern."""
    figure = plt.figure(figsize=(4.0 * len(sections), 8.4), dpi=55)
    largest = max(np.abs(stress.to_array()).max() for _, stress, _ in pushes)
    axes = figure.subplots(points.shape[0], len(sections))
    figure.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99, hspace=0.05, wspace=0.05)
    for push in range(points.shape[0]):
        for case, ((mesh, nodes, _), centre, (strains, stress, polars)) in enumerate(zip(sections, centres, pushes)):
            draw_push(axes[push, case], nodes.batch()[mesh.edges], points[push], strains[push], stress[push], polars,
                      centre, largest)
    return figure
