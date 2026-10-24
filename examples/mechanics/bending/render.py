"""Drawing for thin-walled bending and twisting: the sections' walls with their centroids and shear
centres; each section with its stress at the clamp, its principal axes, its neutral axis, the load and
the way the tip moves; the bent cantilevers beside the straight beam; a load turning or moving across
the tip; and each section pushed along the beam, with the polars of its nodes and its kern.

Sections are seen from beyond the tip, looking back at the clamp: across to the right and up. Walls are
drawn as thick lines along their midlines. The beams are drawn on the GPU as each wall swept along the beam, lit
strips seen along parallel rays, the straight beam a faint ghost beside them. Coordinates are read out here alone:
points through their weight, points at infinity as headings, and lines by the foot of the centroid on
them and their heading."""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, to_rgb, to_rgba
from matplotlib.lines import Line2D
from matplotlib.text import Annotation

from numga import stack
from examples.util.animation import capture
from examples.mechanics.bending import core
from examples.util.wgsl import fitted, image_figure, meshes, show, solids

ga, mv = core.ga, core.mv
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
# The view of the stubby beam, nearly from the side, elevation and azimuth in radians, and how much the beam with its
# slides held rigid hides of what lies behind it.
STUBBY_ELEVATION, STUBBY_AZIMUTH = np.radians([12.0, 20.0])
RIGID_OPACITY = 0.45
# The clamp's square, along the beam, across and up, around the walls at the clamp.
CLAMP = np.array([[0.0, -1.0, -1.0], [0.0, 1.0, -1.0], [0.0, 1.0, 1.0], [0.0, -1.0, 1.0]])
# The colours of the line of shear centres and of the clamp.
SHEAR_LINE, CLAMP_COLOUR = "#0f172a", "#475569"
# The view of the beams, elevation and azimuth in radians, azimuth zero looking along +y; the margin around them
# as a share of the larger side, the pixels across the stubby beam's picture, how much each side of a straight
# beam's wall hides, and the strips' sheen.
ELEVATION, AZIMUTH = np.radians([22.0, -58.0 + 90.0])
MARGIN = 0.1
PIXELS = 520
STRAIGHT_OPACITY = 0.19
GLOSS = 0.1
# The radii of the shear centres' dashes, of the clamp's rods and of the load's shaft, in pixels; the load's head,
# its length a share of the arrow's, and its radius a share of its own length.
DASH_RADIUS, CLAMP_RADIUS, SHAFT_RADIUS = 0.85, 1.2, 1.7
# The radii of the clamp's rods and of the load's shaft in the stubby beam's larger picture, in pixels.
STUBBY_CLAMP_RADIUS, STUBBY_SHAFT_RADIUS = 1.4, 2.0
HEAD_SHARE = 0.25
HEAD_WIDTH = 0.4


def flat(points: core.Point) -> np.ndarray:
    """Points in the section views `[..., 2]`: across and up, any part along the beam projected away."""
    coordinates = points.cast(ga.subspace("ywz wxz xyz")).kernel
    return coordinates[..., :2] / coordinates[..., 2:]


def heading(directions: core.Point) -> np.ndarray:
    """Points at infinity in the section views `[..., 2]`: across and up."""
    return directions.cast(ga.subspace("ywz wxz")).kernel


def space(points: core.Point) -> np.ndarray:
    """Points in the three-dimensional views `[..., 3]`: along the beam, across, and up."""
    return points.cast(ga.subspace("xwy ywz wxz")).kernel


def twist(frames: core.Motor) -> np.ndarray:
    """How far each frame turns a cross-section about the beam, in radians: the turn of the direction
    across the section."""
    turned = heading(frames >> mv.x.dual())                                  # [..., 2]
    return np.arctan2(turned[..., 1], turned[..., 0])


def unit(directions: np.ndarray) -> np.ndarray:
    return directions / np.linalg.norm(directions, axis=-1, keepdims=True)


def arrow(ax, start: np.ndarray, direction: np.ndarray, length: float, colour: str) -> Annotation:
    """An arrow from a start along a direction, a length long, for an animation to move with `aim`."""
    pointer = ax.annotate("", xy=start, xytext=start,
                          arrowprops=dict(arrowstyle="-|>", color=colour, lw=2, shrinkA=0, shrinkB=0))
    aim(pointer, start, direction, length)
    return pointer


def aim(pointer: Annotation, start: np.ndarray, direction: np.ndarray, length: float) -> None:
    """An arrow moved to run from a start along a direction, a length long."""
    pointer.xy = start + length * unit(direction)
    pointer.set_position(start)


def line_ends(line: core.Strain, centre: core.Point, half: float) -> np.ndarray:
    """The ends `[2, 2]` of a line of the section drawn for `half` either side of the foot of `centre` on it."""
    foot = flat((centre | line) ^ line)
    return foot + np.outer([-half, half], unit(heading(line ^ mv.w)))


def line_through(ax, line: core.Strain, centre: core.Point, half: float, **style) -> Line2D:
    """A line of the section, drawn for `half` either side of the foot of `centre` on it."""
    return ax.plot(*line_ends(line, centre, half).T, **style)[0]


def wall_stress(stress: core.Scalar) -> np.ndarray:
    """The stress along the beam at the middle of each piece of each wall `[walls * pieces]`, linear along each wall
    between its ends."""
    values = stress.to_array()                                               # [walls, 2]
    fractions = np.linspace(0, 1, WALL_PIECES + 1)
    middles = (fractions[:-1] + fractions[1:]) / 2
    return (values[:, None, 0] * (1 - middles) + values[:, None, 1] * middles).reshape(-1)


def draw_stress(ax, walls: core.Point, stress: core.Scalar, largest: float) -> LineCollection:
    """The walls as thick lines coloured by the stress along the beam, linear along each wall between
    its ends, for an animation to set the stress of."""
    ends = flat(walls)                                                       # [walls, 2, 2]
    fractions = np.linspace(0, 1, WALL_PIECES + 1)[:, None]                  # [pieces + 1, 1]
    along = ends[:, None, 0] * (1 - fractions) + ends[:, None, 1] * fractions   # [walls, pieces + 1, 2]
    pieces = np.stack([along[:, :-1], along[:, 1:]], axis=-2).reshape(-1, 2, 2)
    lines = LineCollection(pieces, array=wall_stress(stress), cmap=STRESS, norm=Normalize(-largest, largest),
                           linewidths=WALL_WIDTH, capstyle="round")
    ax.add_collection(lines)
    return lines


def frame_section(ax, limit: float) -> None:
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal")
    ax.set_axis_off()


def mark_centres(ax, centre: core.Point, shear: core.Point) -> None:
    """The centroid as a dot, and the shear centre as a ring."""
    ax.plot(*flat(centre), "o", color="#0f172a", markersize=3)
    ax.plot(*flat(shear), "o", markerfacecolor="none", markeredgecolor="#0f172a", markersize=7, markeredgewidth=1.4)


def draw_walls(sections: list[tuple[core.Point, np.ndarray, np.ndarray]], centres: list[core.Point],
               shears: list[core.Point]) -> plt.Figure:
    """The sections' walls along their midlines, each with its centroid and its shear centre."""
    figure, axes = plt.subplots(1, len(sections), figsize=(4 * len(sections), 3.8))
    for ax, (nodes, ends, _), centre, shear in zip(axes, sections, centres, shears):
        ax.add_collection(LineCollection(flat(nodes[ends]), colors=STRAIGHT, linewidths=WALL_WIDTH, capstyle="round"))
        mark_centres(ax, centre, shear)
        frame_section(ax, SECTION_LIMIT)
    figure.tight_layout()
    return figure


def section_limit(tips: core.Point) -> float:
    """Half the width of section views that hold the clamp's section and every moved tip's walls `[..., walls, 2]`
    with a load's arrow beside them."""
    return max(SECTION_LIMIT, np.abs(flat(tips)).max() + ARROW)


def draw_section(ax, walls: core.Point, stress: core.Scalar, principal: core.Strain, centre: core.Point,
                 shear: core.Point, through: core.Point, load: core.Point, tip: core.Point, strain: core.Twist,
                 largest: float, limit: float) -> tuple[LineCollection, LineCollection, Line2D, Annotation]:
    """One section: the stress along the beam over its walls at the clamp, the walls at the tip as the
    tip has moved, its principal axes dashed through the centroid, the neutral axis, and the load's
    direction where it acts on the moved tip. The stress, the tip's walls, the neutral axis and the load's arrow,
    for an animation to move."""
    stresses = draw_stress(ax, walls, stress, largest)
    tip_walls = ax.add_collection(LineCollection(flat(tip), colors=BEAM, linewidths=WALL_WIDTH, capstyle="round"))
    for axis in principal:
        line_through(ax, axis, centre, 2 * limit, color=PRINCIPAL, linestyle="--", linewidth=1.0)
    neutral = line_through(ax, strain, centre, 2 * limit, color=NEUTRAL, linewidth=1.8)
    pointer = arrow(ax, flat(through), heading(load), ARROW, LOAD)
    mark_centres(ax, centre, shear)
    frame_section(ax, limit)
    return stresses, tip_walls, neutral, pointer


def straightened(bent: np.ndarray) -> np.ndarray:
    """The straight beam beside bent ones `[..., pieces + 1, walls, 2, 3]`: the walls at the clamp carried straight
    along the beam, as far along as each cross-section's middle lies along the bent beam."""
    middle = bent.mean(axis=(-3, -2))                                        # [..., pieces + 1, 3]
    steps = np.linalg.norm(np.diff(middle, axis=-2), axis=-1)               # [..., pieces]
    arc = np.concatenate([np.zeros(steps.shape[:-1] + (1,)), np.cumsum(steps, axis=-1)], axis=-1)
    return bent[..., :1, :, :, :] + arc[..., None, None, None] * np.array([1.0, 0.0, 0.0])


def load_tip(through: core.Point, load: core.Point, length: float) -> np.ndarray:
    """Where the drawn load ends `[..., 3]`, a length from where it acts along its direction."""
    return space(through) + length * unit(space(load))


def view(bent: list[np.ndarray], tips: list[np.ndarray], elevation: float, azimuth: float) -> solids.View:
    """A view from an elevation and azimuth framing every bent beam `[..., pieces + 1, walls, 2, 3]` with its straight
    beam, the clamp, and the loads' tips `[..., 3]`."""
    points = [CLAMP] + bent + [straightened(beam) for beam in bent] + tips
    return solids.view([solids.mv.point(batch) for batch in points], 0.0, elevation, azimuth, MARGIN, PIXELS)


def swept(bent: np.ndarray, colour: str, view: solids.View) -> meshes.Samples:
    """Each wall of a bent beam `[pieces + 1, walls, 2, 3]` swept along it as lit strips, as `meshes.samples`."""
    # Each wall between two cross-sections a quad, two triangles, by the indices of its corners.
    index = np.arange(bent[..., 0].size).reshape(bent.shape[:-1])            # [pieces + 1, walls, 2]
    quads = np.stack([index[:-1, :, 0], index[:-1, :, 1], index[1:, :, 1], index[1:, :, 0]], axis=-1)
    triangles = quads[..., [[0, 1, 2], [0, 2, 3]]].reshape(-1, 3)             # [pieces * walls * 2, 3]
    places = solids.mv.point(bent.reshape(-1, 3))
    return meshes.samples(places, triangles, np.array(to_rgb(colour)), meshes.flat(places, triangles), view,
                          face_values=np.zeros(4), colour=meshes.plain, gloss=GLOSS)


def ghost(bent: np.ndarray, view: solids.View) -> solids.Plane:
    """The straight beam beside a bent one `[pieces + 1, walls, 2, 3]`: each wall a plate a pixel thick, from its
    ends at the clamp to its ends at the tip, bounded by planes `[walls, 6]`."""
    straight = solids.mv.point(straightened(bent))                           # [pieces + 1, walls, 2] Point
    return solids.plates(stack([straight[0, :, 0], straight[0, :, 1], straight[-1, :, 1], straight[-1, :, 0]],
                               axis=-1), view.pixel)


def clamped(through: core.Point, load: core.Point, length: float, radii: tuple[float, float],
            view: solids.View) -> list[tuple[solids.Quadric, np.ndarray]]:
    """The clamp, and the tip load where it acts, drawn a length long; the radii of the clamp's rods and of the load's
    shaft in pixels."""
    clamp_radius, shaft_radius = radii
    clamp = solids.mv.point(CLAMP)                                           # [4] Point
    clamp_rods, clamp_joints = solids.edges(clamp, clamp[[1, 2, 3, 0]], clamp_radius * view.pixel)
    shaft, head = solids.arrows(solids.mv.point(space(through)), solids.mv.point(load_tip(through, load, length)),
                                shaft_radius * view.pixel, HEAD_SHARE * length, HEAD_WIDTH * HEAD_SHARE * length)
    load_colour, clamp_colour = (np.array(to_rgb(colour)) for colour in (LOAD, CLAMP_COLOUR))
    return [(clamp_rods, clamp_colour), (clamp_joints, clamp_colour), (shaft[None], load_colour), (head[None], load_colour)]


def draw_beam(walls: core.Point, shears: core.Point, through: core.Point, load: core.Point,
              view: solids.View) -> np.ndarray:
    """One cantilever: each wall swept along the beam as lit strips; the straight beam a faint ghost, each wall a
    translucent plate; the line of shear centres the beam twists about, dashed; the clamp; and the tip load where it
    acts."""
    bent = space(walls)                                                      # [pieces + 1, walls, 2, 3]
    # The line of shear centres in dashes, every other piece of it.
    line = solids.mv.point(space(shears))                                    # [pieces + 1] Point
    dashes = solids.rods(line[:-1:2], line[1::2], DASH_RADIUS * view.pixel)  # [dashes, 3] Quadric
    opaque = [(dashes, np.array(to_rgb(SHEAR_LINE)))] + clamped(through, load, TIP_ARROW, (CLAMP_RADIUS, SHAFT_RADIUS),
                                                                view)
    return solids.draw(opaque, view, np.ones(3), inks=[],
                       translucent=[(ghost(bent, view), np.array(to_rgb(STRAIGHT)), STRAIGHT_OPACITY)],
                       meshes=[(swept(bent, BEAM, view), 1.0)])


def draw_beams(sections: list[tuple[core.Point, np.ndarray, np.ndarray]], centres: list[core.Point],
               shears: list[core.Point], load: core.Point,
               beams: list[tuple[core.Twist, core.Motor, core.Strain, core.Scalar]]) -> plt.Figure:
    """Top, each section with its stress at the clamp; below, each cantilever bent under the load
    through its centroid."""
    figure = plt.figure(figsize=(4.4 * len(sections), 8.6), dpi=110)
    count = len(sections)
    section_axes = [figure.add_subplot(2, count, case + 1) for case in range(count)]
    beam_axes = [figure.add_subplot(2, count, count + case + 1) for case in range(count)]
    figure.subplots_adjust(left=0.01, right=0.99, bottom=0.0, top=0.97, hspace=0.0, wspace=0.05)
    largest = max(np.abs(stress.to_array()).max() for _, _, _, stress in beams)
    walls = [frames[..., None, None] >> nodes[ends] for (nodes, ends, _), (_, frames, _, _) in zip(sections, beams)]
    tips = [load_tip(frames[-1] >> centre, load, TIP_ARROW) for centre, (_, frames, _, _) in zip(centres, beams)]
    # The beams' rooms alike, one view fitted to the first holds them all.
    seen = fitted(view([space(beam) for beam in walls], tips, ELEVATION, AZIMUTH), beam_axes[0])
    limit = max(section_limit(beam[-1]) for beam in walls)
    for section_ax, beam_ax, (nodes, ends, _), centre, shear, (strains, frames, principal, stress), beam in zip(
            section_axes, beam_axes, sections, centres, shears, beams, walls):
        draw_section(section_ax, nodes[ends], stress, principal, centre, shear, frames[-1] >> centre, load,
                     frames[-1] >> nodes[ends], strains[0], largest, limit)
        show(beam_ax, draw_beam(beam, frames >> shear, frames[-1] >> centre, load, seen), seen)
    return figure


def draw_stubby(section: tuple[core.Point, np.ndarray, np.ndarray], frames: list[core.Motor], centre: core.Point,
                load: core.Point) -> plt.Figure:
    """A stubby beam nearly from the side, bent with the walls' shear stiffness, and translucent with the slides held
    rigid, beside the straight beam, with the clamp and the load at the tip."""
    nodes, ends, _ = section
    bent = [space(placed[:, None, None] >> nodes[ends]) for placed in frames]   # [pieces + 1, walls, 2, 3] each
    through = frames[0][-1] >> centre
    seen = view(bent, [load_tip(through, load, ARROW)], STUBBY_ELEVATION, STUBBY_AZIMUTH)
    held = clamped(through, load, ARROW, (STUBBY_CLAMP_RADIUS, STUBBY_SHAFT_RADIUS), seen)
    return image_figure(solids.draw(held, seen, np.ones(3),
                                    inks=[],
                                    translucent=[(ghost(bent[0], seen), np.array(to_rgb(STRAIGHT)), STRAIGHT_OPACITY)],
                                    meshes=[(swept(bent[0], BEAM, seen), 1.0), (swept(bent[1], PRINCIPAL, seen), RIGID_OPACITY)]))


def animate_loads(section: tuple[core.Point, np.ndarray, np.ndarray], centre: core.Point, shear: core.Point,
                  through: core.Point, loads: core.Point,
                  beam: tuple[core.Twist, core.Motor, core.Strain, core.Scalar]) -> Iterator[np.ndarray]:
    """Each frame of a changing tip load, as `sized_loads` draws it at 100 pixels to the inch."""
    return sized_loads(section, centre, shear, through, loads, beam, 100)


def sized_loads(section: tuple[core.Point, np.ndarray, np.ndarray], centre: core.Point, shear: core.Point,
                through: core.Point, loads: core.Point, beam: tuple[core.Twist, core.Motor, core.Strain, core.Scalar],
                dpi: int) -> Iterator[np.ndarray]:
    """Each frame of a changing tip load, through points `through` in directions `loads`, either batched
    over the frames, at `dpi` pixels to the inch: the bent beam, and its section's stress at the clamp with the tip's
    cross-section."""
    nodes, ends, _ = section
    strains, frames, principal, stress = beam
    count = frames.shape[0]
    through, loads = through.broadcast_to((count,)), loads.broadcast_to((count,))
    walls = frames[..., None, None] >> nodes[ends]                           # [frames, pieces + 1, walls, 2] Point
    largest = np.abs(stress.to_array()).max()
    limit = section_limit(walls[:, -1])
    moved = frames[:, -1] >> through                                         # [frames] Point
    # The figure and the section's still parts drawn once; the beam's picture, the stress, the tip's walls, the
    # neutral axis and the load's arrow set for each frame.
    figure = plt.figure(figsize=(9.0, 4.6), dpi=dpi)
    beam_ax, section_ax = figure.add_subplot(1, 2, 1), figure.add_subplot(1, 2, 2)
    figure.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.98, wspace=0.05)
    # One camera framing every frame's beam.
    seen = fitted(view([space(walls)], [load_tip(frames[:, -1] >> through, loads, TIP_ARROW)], ELEVATION, AZIMUTH),
                  beam_ax)
    image = show(beam_ax, np.ones((*seen.shape, 3)), seen)
    stresses, tip_walls, neutral, pointer = draw_section(section_ax, nodes[ends], stress[0], principal, centre, shear,
                                                         moved[0], loads[0], walls[0, -1], strains[0, 0], largest,
                                                         limit)
    for frame in range(count):
        image.set_data(draw_beam(walls[frame], frames[frame] >> shear, moved[frame], loads[frame], seen))
        stresses.set_array(wall_stress(stress[frame]))
        tip_walls.set_segments(flat(walls[frame, -1]))
        neutral.set_data(*line_ends(strains[frame, 0], centre, 2 * limit).T)
        aim(pointer, flat(moved[frame]), heading(loads[frame]), ARROW)
        yield capture(figure)
    plt.close(figure)


def kern(polars: core.Strain, centre: core.Point) -> np.ndarray:
    """The kern as a polygon `[corners, 2]`: the view cut down to the centroid's side of every node's
    polar. Each polar is read as the stretch it gives the fibres, affine across the section."""
    probes = core.point([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])               # [3] Point
    values = (polars[:, None] & (core.beam_direction & probes)).to_array()  # [nodes, 3]
    lines = np.stack([values[:, 1] - values[:, 0], values[:, 2] - values[:, 0], values[:, 0]], axis=-1)
    lines *= np.sign(lines @ np.append(flat(centre), 1.0))[:, None]          # positive on the centroid's side
    polygon = SECTION_LIMIT * np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    for line in lines:
        # Keep each corner on the centroid's side, and where an edge crosses the polar, the crossing.
        side = polygon @ line[:2] + line[2]                                  # [corners]
        kept = []
        for corner, following, here, there in zip(polygon, np.roll(polygon, -1, axis=0), side, np.roll(side, -1)):
            if here >= 0:
                kept.append(corner)
            if here * there < 0:
                kept.append(corner + (following - corner) * here / (here - there))
        polygon = np.array(kept)
    return polygon


def draw_push(ax, walls: core.Point, point: core.Point, strain: core.Strain, stress: core.Scalar,
              polars: core.Strain, centre: core.Point, largest: float) -> None:
    """One section pushed through one point: the stress over its walls, the kern, and the neutral axis,
    the push point's polar."""
    draw_stress(ax, walls, stress, largest)
    ax.fill(*kern(polars, centre).T, facecolor=to_rgba(KERN, 0.3), edgecolor=KERN, linewidth=1.4, zorder=3)
    line_through(ax, strain, centre, 4 * SECTION_LIMIT, color=NEUTRAL, linewidth=1.8)
    ax.plot(*flat(point), "X", color=LOAD, markersize=9, markeredgecolor="white", zorder=4)
    ax.plot(*flat(centre), "o", color="#0f172a", markersize=3, zorder=4)
    frame_section(ax, SECTION_LIMIT)


def draw_pushes(sections: list[tuple[core.Point, np.ndarray, np.ndarray]], centres: list[core.Point],
                points: core.Point, pushes: list[tuple[core.Strain, core.Scalar, core.Strain]]) -> plt.Figure:
    """A row per push point and a column per section: the section pushed along the beam through the
    point, with its kern."""
    figure = plt.figure(figsize=(4.0 * len(sections), 8.4), dpi=110)
    largest = max(np.abs(stress.to_array()).max() for _, stress, _ in pushes)
    for push in range(points.shape[0]):
        for case, ((nodes, ends, _), centre, (strains, stress, polars)) in enumerate(zip(sections, centres, pushes)):
            ax = figure.add_subplot(points.shape[0], len(sections), push * len(sections) + case + 1)
            draw_push(ax, nodes[ends], points[push], strains[push], stress[push], polars, centre, largest)
    figure.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99, hspace=0.05, wspace=0.05)
    return figure
