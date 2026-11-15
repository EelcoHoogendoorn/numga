"""Section and cantilever rendering draws thin-walled cross-sections with centroids, shear centres,
clamp stresses, principal and neutral axes, tip deflections, and kerns alongside GPU-lit swept walls
of the bent and reference cantilevers.

Section views look back along the beam from beyond the tip, reading horizontal and vertical
positions by pairing points with the transverse planes `x` and `y` over unit weight `w`. Walls are
drawn as thick strokes along their midlines, while axes, loads, and kerns overlay strokes, markers,
and filled polygons. Each cantilever sweeps its walls along the piece frames on the GPU beside a
translucent reference beam, rotated from the beam frame into the camera frame by one motor.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, to_rgb

from numga import concatenate, stack
from examples.util.animation import capture
from examples.mechanics.bending import core
from numga.topology import Topology
from numga.render.drawing import Draw
from numga.render import Camera, contours, drawing, meshes, simplices, solids
from examples.util.panels import fitted, overlay, panel, show

mv = core.mv
# Palette for bent walls, straight reference walls, load arrows, neutral axes, principal axes, and kerns.
beam_colour, straight_colour, load_colour = "#0369a1", "#cbd5e1", "#f97316"
neutral_colour, principal_colour, kern_colour = "#16a34a", "#64748b", "#a855f7"
stress_cmap = "RdBu_r"
wall_width = 5.0
n_wall_pieces = 12
section_half_width = 1.2
section_arrow_length, tip_arrow_length = 0.45, 2.4
# Unit-square corners `(x, y)` framing the clamp and initial kern polygon.
unit_square = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
shear_line_colour, clamp_colour = "#0f172a", "#475569"
# Camera elevation and azimuth in radians, framing margin, render resolution, reference opacity, and gloss.
view_elevation, view_azimuth = np.radians([22.0, -58.0 + 90.0])
view_margin = 0.1
n_view_pixels = 520
straight_opacity = 0.19
strip_gloss = 0.1
# Point-sized stroke widths, dash patterns, and marker dimensions in the section views.
principal_width, principal_dashes = 1.0, (3.7, 1.6)
neutral_width, load_width, load_chevron, kern_width = 1.8, 2.0, 8.0, 1.4
dot_side, ring_side, ring_width, push_side, push_width, push_rim = 3.0, 7.0, 1.4, 9.0, 2.4, 1.0
kern_opacity = 0.3
# Pixel radii and shaft dimensions in the 3D cantilever views.
dash_radius, clamp_radius = 0.85, 1.2
shaft_width, shaft_chevron = 3.4, 13.6
# Motor rotating the beam frame (`z` axial, `x` across, `y` up) into the camera frame (`x` axial, `y` across, `z` up).
drawing_turn = (mv.yz * (-np.pi / 4)).exp() * (mv.zx * (-np.pi / 4)).exp()  # [] Motor+
# Intrinsic unit antipseudoscalar converting an extrinsic meet at infinity into an intrinsic direction point.
unit_volume = mv.antipseudoscalar([1])                                      # [] Antipseudoscalar-


# --- plumbing -------------------------------------------------------------------------
def section_coordinates(points: core.Point) -> np.ndarray:
    """Pairing `points` with the transverse planes `x` and `y` over weight `w` reads out `(x, y)`
    section coordinates (`[..., 2]`), ignoring axial offsets."""
    batch_points = points.batch()
    return ((stack((mv.x, mv.y)) & batch_points[..., None]) / (mv.w & batch_points)[..., None]).to_array()


def offset_point(start: core.Point, direction: core.Point, distance: float | np.ndarray) -> core.Point:
    """Advancing unit-weight `start` along `direction` by `distance` produces the displaced point
    (`[...] Point-`)."""
    return start + direction / direction.dual().norm() * distance           # [...] Point-


def line_endpoints(line: core.Strain, centre: core.Point, half_length: float) -> core.Point:
    """Projecting `centre` orthogonally onto `line` and stepping `half_length` either way along its
    ideal point gives the two segment endpoints (`[2] Point-`)."""
    foot = ((centre | line) ^ line).normalized()                            # [] Point-
    direction = (line ^ mv.w) * unit_volume                                 # [] Point-
    return offset_point(foot, direction, half_length * np.array([-1.0, 1.0]))  # [2] Point-


def wall_piece_stresses(end_stress: core.Scalar) -> np.ndarray:
    """Interpolating wall-end stresses (`[2] Scalar+[E]`) to the midpoints of `n_wall_pieces`
    sub-segments per wall gives `[E * n_wall_pieces]` colour values."""
    tails, heads = end_stress.to_array()                                    # [E], [E]
    fractions = np.linspace(0, 1, n_wall_pieces + 1)                        # [n_wall_pieces + 1]
    midpoints = (fractions[:-1] + fractions[1:]) / 2                        # [n_wall_pieces]
    return (tails[:, None] * (1 - midpoints) + heads[:, None] * midpoints).reshape(-1)


def wall_strokes(wall_ends: core.Point, colours: np.ndarray, scale: float, camera: Camera) -> np.ndarray:
    """Wall segments (`[E, 2] Point-`) rasterize as thick strokes in `colours`."""
    return simplices.strokes(wall_ends.reshape(-1), Topology.chains([2] * wall_ends.shape[0]), colours,
                             wall_width * scale, camera)


def stress_strokes(wall_ends: core.Point, end_stress: core.Scalar, max_stress: float, scale: float,
                   camera: Camera) -> np.ndarray:
    """Subdividing each wall (`[E, 2] Point-`) into `n_wall_pieces` pieces colours the midline linearly
    by axial stress."""
    n_walls = wall_ends.shape[0]
    fractions = np.linspace(0, 1, n_wall_pieces + 1)                        # [n_wall_pieces + 1]
    subdivided = wall_ends[:, None, 0] * (1 - fractions) + wall_ends[:, None, 1] * fractions  # [n_walls, n_wall_pieces + 1] Point-
    colours = plt.get_cmap(stress_cmap)(Normalize(-max_stress, max_stress)(wall_piece_stresses(end_stress)))[:, :3]
    return simplices.strokes(subdivided.reshape(-1), Topology.chains([n_wall_pieces + 1] * n_walls), colours,
                             wall_width * scale, camera)


def axis_stroke(line: core.Strain, centre: core.Point, half_length: float, colour: str, width: float,
                camera: Camera, dashes: tuple[float, float] = (1.0, 0.0)) -> np.ndarray:
    """One section axis `line` renders for `half_length` either side of the foot of `centre` at stroke
    `width`."""
    return simplices.strokes(line_endpoints(line, centre, half_length), Topology.chains([2]),
                             np.array(to_rgb(colour)), width, camera, dashes=dashes, cap=simplices.Cap.BUTT)


def section_panel(ax, limit: float) -> tuple[Camera, float]:
    """Framing `ax` over `[-limit, limit]` in both directions returns its transverse `(x, y, w)`
    `Camera` and pixels-per-point scale."""
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal")
    ax.set_axis_off()
    return panel(ax, mv.x, mv.y, mv.w), ax.figure.dpi / 72


def centre_markers(centre: core.Point, shear: core.Point, scale: float, camera: Camera) -> list[np.ndarray]:
    """Drawing a ring at `shear` beneath a solid dot at `centre` marks both points even when they
    coincide."""
    ink = np.array(to_rgb("#0f172a"))
    return [simplices.markers(shear[None], ink, (ring_side + ring_width) / 2 * scale, camera),
            simplices.markers(shear[None], np.ones(3), (ring_side - ring_width) / 2 * scale, camera),
            simplices.markers(centre[None], ink, dot_side / 2 * scale, camera)]


def section_row(sections: list[tuple[core.Topology, core.Point, core.Scalar]]
                ) -> tuple[plt.Figure, list, list[Camera], float]:
    """One row of section axes creates a shared figure, axis list, cameras, and pixels-per-point scale."""
    figure, axes = plt.subplots(1, len(sections), figsize=(4 * len(sections), 3.8), dpi=50)
    for ax in axes:
        ax.set_axis_off()
    figure.tight_layout()
    panels = [section_panel(ax, section_half_width) for ax in axes]
    return figure, axes, [camera for camera, _ in panels], panels[0][1]


def draw_sections(sections: list[tuple[core.Topology, core.Point, core.Scalar]]) -> plt.Figure:
    """Midline strokes display the thin-walled cross-sections side by side."""
    figure, axes, cameras, scale = section_row(sections)
    for ax, camera, (topology, nodes, _) in zip(axes, cameras, sections):
        overlay(ax, [simplices.strokes(nodes.batch(), topology, np.array(to_rgb(straight_colour)),
                                       wall_width * scale, camera)])
    return figure


def draw_walls(sections: list[tuple[core.Topology, core.Point, core.Scalar]], centres: list[core.Point],
               shears: list[core.Point]) -> plt.Figure:
    """Midline strokes display each cross-section together with its centroid dot and shear-centre ring."""
    figure, axes, cameras, scale = section_row(sections)
    for ax, camera, (topology, nodes, _), centre, shear in zip(axes, cameras, sections, centres, shears):
        overlay(ax, [simplices.strokes(nodes.batch(), topology, np.array(to_rgb(straight_colour)),
                                       wall_width * scale, camera),
                     *centre_markers(centre, shear, scale, camera)])
    return figure


def section_view_limit(tip_walls: core.Point) -> float:
    """Pairing displaced `tip_walls` (`[..., E, 2] Point-`) with `x` and `y` over `w` sets a half-width
    that encloses both clamp and deflected tip sections plus a load arrow."""
    flat_ends = tip_walls.batch().reshape(-1)                               # [...] Point-
    readings = (stack((mv.x, mv.y))[:, None] & flat_ends) / (mv.w & flat_ends)  # [2, ...] Scalar+
    return max(section_half_width, float(readings.abs().max().to_array()) + section_arrow_length)


def section_layers(wall_ends: core.Point, clamp_stress: core.Scalar, principal: core.Strain,
                   centre: core.Point, shear: core.Point, load_point: core.Point, load_direction: core.Point,
                   tip_walls: core.Point, clamp_strain: core.Twist, max_stress: float, limit: float,
                   scale: float, camera: Camera) -> list[np.ndarray]:
    """Clamp wall stresses, displaced tip walls, principal axes, neutral axis, load arrow, centroid,
    and shear centre stack into RGBA layers for one section panel."""
    layers = [stress_strokes(wall_ends, clamp_stress, max_stress, scale, camera),
              wall_strokes(tip_walls, np.array(to_rgb(beam_colour)), scale, camera)]
    layers += [axis_stroke(axis, centre, 2 * limit, principal_colour, principal_width * scale, camera,
                           tuple(np.array(principal_dashes) * principal_width * scale)) for axis in principal]
    layers.append(axis_stroke(clamp_strain.cast(core.Strain), centre, 2 * limit, neutral_colour,
                              neutral_width * scale, camera))
    arrow_tip = offset_point(load_point, load_direction, section_arrow_length)  # [] Point-
    layers.append(simplices.strokes(stack([load_point, arrow_tip]), Topology.chains([2]),
                                    np.array(to_rgb(load_colour)), load_width * scale, camera,
                                    chevrons=np.ones(1), chevron=load_chevron * scale))
    return layers + centre_markers(centre, shear, scale, camera)


def reference_beam(bent_walls: core.Point) -> core.Point:
    """Advancing clamp walls straight along `beam_direction` by the cumulative arc step between section
    centres builds the unbent reference walls (`[..., n_pieces + 1, E, 2] Point-`)."""
    middles = bent_walls.mean(axis=-1).mean(axis=-1)                        # [...] Point-
    steps = (middles[..., :-1] & middles[..., 1:]).norm()                   # [...] Scalar+
    arc_lengths = concatenate([0 * steps[..., :1], steps.cumsum(axis=-1)], axis=-1)  # [...] Scalar+
    return bent_walls[..., :1, :, :] + core.beam_direction * arc_lengths[..., None, None]  # [...] Point-


def beam_view(bent_beams: list[core.Point], arrow_tips: list[core.Point], elevation: float,
              azimuth: float) -> solids.View:
    """Framing every bent beam, its straight reference, the clamp square, and the load arrows builds
    one shared `solids.View`."""
    batches = ([core.section_point(unit_square)] + bent_beams
               + [reference_beam(beam) for beam in bent_beams] + arrow_tips)
    return solids.view([drawing_turn >> batch for batch in batches], 0.0, elevation, azimuth,
                       view_margin, n_view_pixels)


def swept_walls(bent_walls: core.Point, colour: str, view: solids.View) -> Draw:
    """Triangulating each wall quad between adjacent cross-sections renders lit wall strips on the GPU."""
    n_sections, n_walls, n_ends = bent_walls.shape
    index = np.arange(n_sections * n_walls * n_ends).reshape(bent_walls.shape)  # [n_sections, n_walls, 2]
    quads = np.stack([index[:-1, :, 0], index[:-1, :, 1], index[1:, :, 1], index[1:, :, 0]], axis=-1)
    triangles = quads[..., [[0, 1, 2], [0, 2, 3]]].reshape(-1, 3)           # [(n_sections - 1) * n_walls * 2, 3]
    places = drawing_turn >> bent_walls.reshape(-1)                         # [...] Point-
    return meshes.samples(places, triangles, np.array(to_rgb(colour)), meshes.flat(places, triangles), view,
                          face_values=np.zeros(4), colour=meshes.plain, gloss=strip_gloss)


def ghost_plates(bent_walls: core.Point, view: solids.View) -> solids.Plane:
    """Extruding clamp walls straight to the tip forms one-pixel-thick translucent reference plates."""
    straight = drawing_turn >> reference_beam(bent_walls)                   # [...] Point-
    return solids.plates(stack([straight[0, :, 0], straight[0, :, 1], straight[-1, :, 1], straight[-1, :, 0]],
                               axis=-1), view.pixel)


def clamp_and_load(load_point: core.Point, load_direction: core.Point, arrow_length: float,
                   rod_radius: float, tube_width: float,
                   view: solids.View) -> tuple[list[tuple[solids.Quadric, np.ndarray]], Draw]:
    """Clamp frame rods and the 3D tip-load arrow assemble for compositing in `solids.draw`."""
    clamp_corners = drawing_turn >> core.section_point(unit_square)         # [4] Point-
    tails, heads = (Topology.loops([4]).incidence[1, 0] * clamp_corners.field()).batch()  # [4] Point-, [4] Point-
    clamp_rods, clamp_joints = solids.edges(tails, heads, rod_radius * view.pixel)
    arrow_tip = drawing_turn >> offset_point(load_point, load_direction, arrow_length)  # [] Point-
    arrow = simplices.tubes(stack([drawing_turn >> load_point, arrow_tip]), Topology.chains([2]),
                            np.array(to_rgb(load_colour)), tube_width, view, gloss=solids.gloss,
                            chevrons=np.ones(1), chevron=shaft_chevron)
    frame_colour = np.array(to_rgb(clamp_colour))
    return [(clamp_rods, frame_colour), (clamp_joints, frame_colour)], arrow


def draw_beam(bent_walls: core.Point, shear_line: core.Point, load_point: core.Point,
              load_direction: core.Point, view: solids.View) -> np.ndarray:
    """Rendering one cantilever composites the swept walls, translucent reference plates, dashed shear
    axis, clamp frame, and tip-load arrow."""
    turned_shears = drawing_turn >> shear_line                              # [...] Point-
    dashes = solids.rods(turned_shears[:-1:2], turned_shears[1::2], dash_radius * view.pixel)  # [..., 3] Plane+ <- Point-
    clamp, arrow = clamp_and_load(load_point, load_direction, tip_arrow_length, clamp_radius, shaft_width, view)
    opaque = [(dashes, np.array(to_rgb(shear_line_colour)))] + clamp
    return solids.draw(opaque, view, np.ones(3), inks=[],
                       translucent=[(ghost_plates(bent_walls, view), np.array(to_rgb(straight_colour)),
                                     straight_opacity)],
                       meshes=[(swept_walls(bent_walls, beam_colour, view), 1.0), (arrow, 1.0)])


def wall_endpoints(topology: Topology, nodes: core.Point) -> core.Point:
    """Reading wall endpoints off `nodes` stacks each wall's two ends along the trailing axis (`[E, 2] Point-`)."""
    V, E = topology.n_elements
    return stack(tuple((topology.incidence[1, 0] * nodes).batch()), axis=-1)  # [E, 2] Point-


def draw_beams(sections: list[tuple[core.Topology, core.Point, core.Scalar]], centres: list[core.Point],
               shears: list[core.Point], load_direction: core.Point,
               beams: list[tuple[core.Twist, core.Motor, core.Strain, core.Scalar]]) -> plt.Figure:
    """Clamp stresses and tip cross-sections appear along the top row above the corresponding 3D bent
    cantilevers along the bottom row."""
    n_sections = len(sections)
    figure = plt.figure(figsize=(4.4 * n_sections, 8.6), dpi=55)
    section_axes = [figure.add_subplot(2, n_sections, index + 1) for index in range(n_sections)]
    beam_axes = [figure.add_subplot(2, n_sections, n_sections + index + 1) for index in range(n_sections)]
    figure.subplots_adjust(left=0.01, right=0.99, bottom=0.0, top=0.97, hspace=0.0, wspace=0.05)
    max_stress = max(float(clamp_stress.abs().max().to_array()) for _, _, _, clamp_stress in beams)
    walls = [wall_endpoints(topology, nodes) for topology, nodes, _ in sections]
    bent_beams = [frames[..., None, None] >> ends for ends, (_, frames, _, _) in zip(walls, beams)]
    arrow_tips = [offset_point(frames[-1] >> centre, load_direction, tip_arrow_length)
                  for centre, (_, frames, _, _) in zip(centres, beams)]
    shared_view = fitted(beam_view(bent_beams, arrow_tips, view_elevation, view_azimuth), beam_axes[0])
    limit = max(section_view_limit(bent[-1]) for bent in bent_beams)
    for section_ax, beam_ax, ends, centre, shear, (strains, frames, principal, clamp_stress), bent in zip(
            section_axes, beam_axes, walls, centres, shears, beams, bent_beams):
        camera, scale = section_panel(section_ax, limit)
        overlay(section_ax, section_layers(ends, clamp_stress, principal, centre, shear,
                                           frames[-1] >> centre, load_direction, bent[-1], strains[0],
                                           max_stress, limit, scale, camera))
        show(beam_ax, draw_beam(bent, frames >> shear, frames[-1] >> centre, load_direction, shared_view),
             shared_view)
    return figure


def animate_loads(section: tuple[core.Topology, core.Point, core.Scalar], centre: core.Point,
                  shear: core.Point, states: Iterable[tuple[core.Point, core.Point, core.Twist,
                                                            core.Motor, core.Strain, core.Scalar]]
                  ) -> Iterator[np.ndarray]:
    """Yields 50-dpi RGB frames (`[rows, columns, 3]`) for each streamed cantilever load state."""
    return sized_loads(section, centre, shear, 50, states)


def sized_loads(section: tuple[core.Topology, core.Point, core.Scalar], centre: core.Point,
                shear: core.Point, dpi: int,
                states: Iterable[tuple[core.Point, core.Point, core.Twist, core.Motor,
                                       core.Strain, core.Scalar]]) -> Iterator[np.ndarray]:
    """Yields `dpi`-scaled RGB frames (`[rows, columns, 3]`) pairing the 3D bent cantilever with its
    clamp-stress and tip-deflection section view across `states`."""
    collected = list(states)
    n_frames = len(collected)
    topology, nodes, _ = section
    walls = wall_endpoints(topology, nodes)
    load_points = stack([state[0] for state in collected])                  # [n_frames] Point-
    load_directions = stack([state[1] for state in collected])              # [n_frames] Point-
    strains = stack([state[2] for state in collected])                      # [n_frames, ...] Twist+
    frames = stack([state[3] for state in collected])                       # [n_frames, ...] Motor+
    principal = collected[0][4]                                             # [2] Strain+
    clamp_stresses = stack([state[5] for state in collected])               # [n_frames, 2] Scalar+[E]
    bent_walls = frames[..., None, None] >> walls                           # [n_frames, ...] Point-
    max_stress = float(clamp_stresses.abs().max().to_array())
    limit = section_view_limit(bent_walls[:, -1])
    moved_points = frames[:, -1] >> load_points                             # [n_frames] Point-
    figure = plt.figure(figsize=(9.0, 4.6), dpi=dpi)
    beam_ax, section_ax = figure.add_subplot(1, 2, 1), figure.add_subplot(1, 2, 2)
    figure.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.98, wspace=0.05)
    shared_view = fitted(beam_view([bent_walls], [offset_point(moved_points, load_directions, tip_arrow_length)],
                                   view_elevation, view_azimuth), beam_ax)
    beam_image = show(beam_ax, np.ones((*shared_view.shape, 3)), shared_view)
    camera, scale = section_panel(section_ax, limit)
    section_image = overlay(section_ax, [np.zeros(camera.shape + (4,))])
    for index in range(n_frames):
        beam_image.set_data(draw_beam(bent_walls[index], frames[index] >> shear, moved_points[index],
                                      load_directions[index], shared_view))
        section_image.set_data(drawing.composited(section_layers(
            walls, clamp_stresses[index], principal, centre, shear, moved_points[index],
            load_directions[index], bent_walls[index, -1], strains[index, 0], max_stress, limit, scale, camera)))
        yield capture(figure)
    plt.close(figure)


def kern_polygon(polars: core.Strain, centre: core.Point) -> core.Point:
    """Clipping the view square to the centroid's side of every node polar (`[V] Strain+`) returns the
    convex kern polygon's vertices (`[n_corners] Point-`)."""
    corners = core.section_point(section_half_width * unit_square)          # [4] Point-
    centroid_stretch = polars & (core.beam_direction & centre)              # [...] Scalar+
    signs = np.asarray(centroid_stretch > 0, dtype=int) - np.asarray(centroid_stretch < 0, dtype=int)
    for polar, sign in zip(polars, signs):
        n_edges = len(corners)
        tails, heads = (Topology.loops([n_edges]).incidence[1, 0] * corners.field()).batch()  # [n_edges] Point-, [n_edges] Point-
        tail_stretches = sign * (polar & (core.beam_direction & tails))     # [n_edges] Scalar+
        head_stretches = sign * (polar & (core.beam_direction & heads))     # [n_edges] Scalar+
        kept = []
        for tail, head, tail_val, head_val in zip(tails, heads, tail_stretches, head_stretches):
            if tail_val >= 0:
                kept.append(tail)
            if tail_val * head_val < 0:
                kept.append((head * tail_val - tail * head_val) / (tail_val - head_val))
        corners = stack(kept)
    return corners


def draw_push(ax, wall_ends: core.Point, push_point: core.Point, strain: core.Strain,
              end_stress: core.Scalar, polars: core.Strain, centre: core.Point, max_stress: float) -> None:
    """One section panel displays wall stresses, the shaded kern polygon, the neutral axis, and a cross
    at `push_point`."""
    camera, scale = section_panel(ax, section_half_width)
    corners = kern_polygon(polars, centre)                                  # [...] Point-
    n_corners = len(corners)
    fan = Topology(n_corners, np.stack([np.zeros(n_corners - 2, dtype=int), np.arange(1, n_corners - 1),
                                        np.arange(2, n_corners)], axis=-1))
    loop = Topology.loops([n_corners])
    fill_colour = np.array(to_rgb(kern_colour))
    arm_rotors = (mv.xy * (np.pi / 8 * np.array([1.0, -1.0]))).exp()        # [2] Motor+
    arm_offsets = (arm_rotors >> mv.x.dual()) * (push_side / 2 * scale * camera.pixel)  # [2] Point-
    cross_ends = concatenate([push_point - arm_offsets, push_point + arm_offsets])  # [4] Point-
    cross_topology = Topology(4, np.array([[0, 2], [1, 3]]))
    overlay(ax, [
        stress_strokes(wall_ends, end_stress, max_stress, scale, camera),
        contours.picture(corners, fan, np.zeros(n_corners), fill_colour, camera, opacity=kern_opacity),
        simplices.strokes(corners, loop, fill_colour, kern_width * scale, camera),
        axis_stroke(strain, centre, 4 * section_half_width, neutral_colour, neutral_width * scale, camera),
        simplices.strokes(cross_ends, cross_topology, np.ones(3), (push_width + 2 * push_rim) * scale, camera),
        simplices.strokes(cross_ends, cross_topology, np.array(to_rgb(load_colour)), push_width * scale, camera),
        centre_markers(centre, centre, scale, camera)[-1]])


def draw_pushes(sections: list[tuple[core.Topology, core.Point, core.Scalar]], centres: list[core.Point],
                push_points: core.Point, pushes: list[tuple[core.Strain, core.Scalar, core.Strain]]) -> plt.Figure:
    """Rows of `push_points` and columns of `sections` compare axial compression inside and outside
    each section's kern."""
    n_pushes = push_points.shape[0]
    n_sections = len(sections)
    figure = plt.figure(figsize=(4.0 * n_sections, 8.4), dpi=55)
    max_stress = max(float(end_stress.abs().max().to_array()) for _, end_stress, _ in pushes)
    axes = figure.subplots(n_pushes, n_sections)
    figure.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99, hspace=0.05, wspace=0.05)
    walls = [wall_endpoints(topology, nodes) for topology, nodes, _ in sections]
    for push_index in range(n_pushes):
        for case_index, (ends, centre, (strains, stresses, polars)) in enumerate(zip(walls, centres, pushes)):
            draw_push(axes[push_index, case_index], ends, push_points[push_index], strains[push_index],
                      stresses[push_index], polars, centre, max_stress)
    return figure
