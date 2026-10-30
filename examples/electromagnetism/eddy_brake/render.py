"""Current paths in a conducting disc beneath a local magnetic field."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.artist import Artist
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.path import Path
from matplotlib.tri import Triangulation

from numga import concatenate, stack, where
from numga.extensor import Extensor
from numga.sparse import SparseExtensor, spdiag
from examples.util.animation import capture
from examples.electromagnetism.eddy_brake import core

Mesh, as_scalar = core.Mesh, core.as_scalar

image_samples = 256
animation_dpi = 90
fibre_reach = 0.94
fibre_colour = (0.16, 0.43, 0.56, 0.8)
fibre_width = 1.35
current_gamma = 0.35
streamline_gamma = 0.25
current_levels = 10
contour_passes = 2
arrow_phase = (np.sqrt(5) - 1) / 2
current_arrow_length = 0.035
# The arc of the magnet's sense of turning, just outside the dashed rim of its width, as a share of that width: where
# it starts around the rim, how far it runs, and the samples along it.
field_arc_reach = 1.3
field_arc_start, field_arc_span = np.radians(-60.0), np.radians(240.0)
n_field_arc = 24
magnet_colour = "#334155"
heat_gamma = 0.3
current_colours = LinearSegmentedColormap.from_list("current", ("#f8f5ed", "#edb15f", "#c95832"))
heat_colours = LinearSegmentedColormap.from_list("heat", ("#fbf9f4", "#e58b3a", "#8f2d1b"))
mode_colours = LinearSegmentedColormap.from_list("mode", ("#2b6c8f", "#f8f5ed", "#c95832"))
mode_levels = 9
# How many times over the braking currents' colour scale and contour values take the first state's: once, as the drag
# only slows the disc; a later state's larger current widens them.
current_room = 1.0
# The tops of the fixed colour scales of the runs that start without current or heat, the most the notebook's runs
# reach: the current of the discs spun up from rest, in A/m, and the heat per area, in J/m², of the braking discs and of
# the discs spun up.
induction_current_peak = 8.55e5
braking_heat_peak = 1.61
induction_heat_peak = 1.94e4


# --- plumbing -------------------------------------------------------------------------
def planar(directions: core.Point) -> np.ndarray:
    """Directions in the disc read by pairing with the slices across x and y, on the last axis."""
    return (stack((core.mv.x, core.mv.y)) & directions[..., None]).to_array()


def scalars(values: core.Scalar) -> np.ndarray:
    """The values of a batch or field of scalars, for drawing."""
    return values.batch().to_array()


def magnet(ax: plt.Axes, centre: core.Point, width: float, field: core.Bivector) -> None:
    """The magnet's field, a plane of space, as a patch in that plane: its width dashed, and an arc around it with an
    arrowhead for the sense the field turns in, read by pairing the field with the patch of the disc's plane."""
    ax.add_patch(Circle(planar(centre), width, facecolor="none", edgecolor=magnet_colour,
                        linewidth=0.75, linestyle=(0, (3, 3)), zorder=4))
    handedness = field & core.mv.xy.dual()
    sense = np.asarray(handedness > 0, dtype=int) - np.asarray(handedness < 0, dtype=int)
    angles = field_arc_start + sense * np.linspace(0, field_arc_span, n_field_arc)
    # The arc: a point on the rim along x, turned about the magnet's centre through the angles.
    turns = (core.mv.xy * (-angles / 2)).exp()                               # [arc] Rotor+
    arc = centre + (turns >> (core.mv.x.dual() * (field_arc_reach * width)))  # [arc] Point-
    ax.add_patch(FancyArrowPatch(path=Path(planar(arc)), arrowstyle="-|>", mutation_scale=8, shrinkA=0, shrinkB=0,
                                 linewidth=0.75, color=magnet_colour, zorder=4))


def parallel_fibres(radius: float, curves: int, samples: int) -> core.Point:
    """Parallel chords marking the direction of straight body-fixed fibres."""
    offsets = np.linspace(-fibre_reach, fibre_reach, curves)
    fractions = np.linspace(-1, 1, samples)
    half_lengths = radius * np.sqrt(1 - offsets ** 2)
    return (core.mv.x.dual() * half_lengths[:, None] * fractions[None, :]
            + core.mv.y.dual() * radius * offsets[:, None])


def radial_fibres(inner_radius: float, outer_radius: float, curves: int, samples: int) -> core.Point:
    """Radial fibre guides between the isotropic centre and rim bands."""
    rays = 2 * curves
    angles = np.linspace(0, 2 * np.pi, rays, endpoint=False)
    distances = np.linspace(inner_radius, outer_radius, samples)
    directions = (core.mv.xy * (-angles / 2)).exp() >> core.mv.x.dual()    # [rays] Point-
    return directions[:, None] * distances[None, :]


def circular_fibres(inner_radius: float, outer_radius: float, curves: int, samples: int) -> core.Point:
    """Closed fibre rings between the isotropic centre and rim bands."""
    radii = np.linspace(inner_radius, outer_radius, curves)
    angles = np.linspace(0, 2 * np.pi, samples)
    directions = (core.mv.xy * (-angles / 2)).exp() >> core.mv.x.dual()    # [samples] Point-
    return directions[None, :] * radii[:, None]


def rotation_marks(orientations: core.Rotor, radius: float) -> np.ndarray:
    """Material rim positions, `[..., 2]`, one for each orientation `[...]`."""
    return planar(orientations >> (core.mv.x.dual() * (fibre_reach * radius)))


def rotation_marker(ax: plt.Axes) -> PathCollection:
    """A fluorescent rim knob showing the material's rotation."""
    return ax.scatter([], [], s=180, c="#ff00c8", edgecolors="white", linewidths=1.5,
                      zorder=6, clip_on=False)


def at_vertices(mesh: Mesh, areas: core.Scalar, field: Extensor) -> Extensor:
    """Face fields averaged at each vertex, weighted by the faces' areas `Scalar+[faces]`, for drawing `[...] X[faces]`
    as `[...] X[vertices]`."""
    third = mesh.incidence[2, 0].mean(axis=0)
    return (~third * (areas * field)) / (~third * areas)


def fibre_colours(fibres: tuple[core.Point, ...], shown: np.ndarray) -> np.ndarray:
    """A colour for every guide curve of every family on every panel, `[cases, curves, 4]`: each family
    is drawn on the panels that show it, `shown` `[cases, families]`."""
    family = np.concatenate([np.full(len(guides), index) for index, guides in enumerate(fibres)])
    return np.concatenate((np.broadcast_to(fibre_colour[:3], shown.shape[:1] + family.shape + (3,)),
                           (fibre_colour[3] * shown[:, family])[..., None]), axis=-1)


def disc_panel(figure: plt.Figure, index: int, count: int, extent: float, centre: core.Point,
               width: float, field: core.Bivector,
               colours: np.ndarray) -> tuple[plt.Axes, LineCollection, PathCollection, LineCollection]:
    """One panel of a row of rotating discs: fibre guides, rim knob, rim outline and the magnet."""
    ax = figure.add_axes(((index + 0.035) / count, 0.03, 0.93 / count, 0.86))
    threads = LineCollection([], colors=colours, linewidths=fibre_width, zorder=2)
    ax.add_collection(threads)
    marker = rotation_marker(ax)
    outline = LineCollection([], colors="#746b60", linewidths=0.8, zorder=3)
    ax.add_collection(outline)
    magnet(ax, centre, width, field)
    ax.set(xlim=(-extent, extent), ylim=(-extent, extent), aspect="equal")
    ax.set_axis_off()
    return ax, threads, marker, outline


def rim(mesh: Mesh, vertices: core.Point) -> np.ndarray:
    """The rim's edges as segments to draw, `[rim, 2, 2]`: each edge's tail and head."""
    tails, heads = mesh.reading(mesh.boundary_edges, 1) * (mesh.incidence[1, 0] * vertices)   # Point-[rim], Point-[rim]
    return np.stack((planar(tails.batch()), planar(heads.batch())), axis=-2)


def corner_gradients(mesh: Mesh, vertices: core.Point) -> core.Vector:
    """The gradients of each face's corner hats, `[corners] Vector+[faces]`: a hat rises from zero along the edge facing
    its corner to one at the corner, so its gradient is the plane through that edge and time, over twice the face's
    signed area."""
    sides = mesh.boundary[2, 1] * (mesh.d[0] * vertices)                    # [corners] Point-[faces]
    double_areas = (sides[0] & sides[1]) & core.time                        # Scalar-[faces]
    return (core.time & sides) / double_areas                               # [corners] Vector+[faces]


def sampling(mesh: Mesh, vertices: core.Point, samples: core.Point, holding: np.ndarray) -> SparseExtensor:
    """The map carrying a field on the vertices to samples `[samples] Point-`, linear across each face: a sample's hats
    in the face that holds it, `holding` `[samples]`, weight that face's corners."""
    corners = mesh.incidence[2, 0]                                          # [corners] Scalar+[faces <- vertices]
    reading = mesh.reading(holding, 2)                                      # Scalar+[samples <- faces]
    hats = 1 + ((reading * corner_gradients(mesh, vertices))
                & (samples.field() - reading * (corners * vertices)))       # [corners] Scalar+[samples]
    return (spdiag(hats) * (reading * corners)).sum(axis=0)                 # Scalar+[samples <- vertices]


def level_loops(mesh: Mesh, vertices: core.Point, stream: core.Scalar,
                levels: np.ndarray) -> list[tuple[int, core.Point, np.ndarray]]:
    """The level lines of a streamfunction `Scalar+[vertices]` that vanishes on the rim, at the levels `[levels]`:
    closed loops, each with its level's index, its points where it crosses the mesh's edges, `[crossings] Point-`, in
    order with the higher values on its left, and the face its segment from each crossing to the next runs through,
    `[crossings]`."""
    tails, heads = (mesh.incidence[1, 0] * stream).batch()                  # [edges] Scalar+, [edges] Scalar+
    starts = (mesh.incidence[1, 0] * vertices)[0].batch()                   # [edges] Point-
    edges = (mesh.d[0] * vertices).batch()                                  # [edges] Point-
    # A level crosses the edges whose ends lie on either side of it, the share of the edge's rise it takes along it.
    above = stream.batch() > levels[:, None]                                # [levels, vertices]
    crossed = above[:, mesh.edges[:, 0]] != above[:, mesh.edges[:, 1]]     # [levels, edges]
    crossing_levels, crossed_edges = np.nonzero(crossed)                    # [crossings], [crossings]
    shares = (tails[crossed_edges] - levels[crossing_levels]) / (tails - heads)[crossed_edges]   # [crossings] Scalar+
    crossings = starts[crossed_edges] + shares * edges[crossed_edges]       # [crossings] Point-

    # A level parts one corner of a face it crosses from the other two, and runs between the edges at that corner,
    # with the higher values on its left: in a face wound counter-clockwise, from the edge before the corner to the
    # edge after it where the corner lies above, and the other way where it lies below.
    corner_above = above[:, mesh.faces]                                     # [levels, faces, corners]
    lonely = (corner_above != np.roll(corner_above, 1, axis=-1)) & (corner_above != np.roll(corner_above, -1, axis=-1))
    level_idx, face_idx, corner_idx = np.nonzero(lonely)
    after = mesh.face_edges[face_idx, (corner_idx + 1) % 3]
    before = mesh.face_edges[face_idx, (corner_idx + 2) % 3]
    high = corner_above[level_idx, face_idx, corner_idx]
    crossing_idx = np.full(crossed.shape, -1)
    crossing_idx[crossed] = np.arange(crossed.sum())
    entries = crossing_idx[level_idx, np.where(high, before, after)]
    following, faces = np.empty_like(entries), np.empty_like(entries)
    following[entries] = crossing_idx[level_idx, np.where(high, after, before)]
    faces[entries] = face_idx

    # Each loop walked from its first crossing until it closes.
    loops, unvisited = [], np.ones(len(following), dtype=bool)
    for first in range(len(following)):
        if unvisited[first]:
            loop = [first]
            while following[loop[-1]] != first:
                loop.append(following[loop[-1]])
            unvisited[loop] = False
            loops.append((crossing_levels[first], crossings[np.array(loop)], faces[loop]))
    return loops


def smoothed(loop: core.Point, passes: int) -> core.Point:
    """A closed loop `[points] Point-` with its corners cut the given number of times by Chaikin's rule: each side
    replaced by its points a quarter of the way in from either end."""
    for _ in range(passes):
        following = loop[np.roll(np.arange(loop.shape[0]), -1)]             # [points] Point-
        loop = stack((0.75 * loop + 0.25 * following, 0.25 * loop + 0.75 * following), axis=-1).reshape(-1)
    return loop


def periodic_location(distances: core.Scalar, samples: core.Scalar) -> tuple[np.ndarray, core.Scalar, core.Scalar]:
    """Locate scalar distances along a closed sampled curve: segment indices, shares and wrapped distances.

    NumPy supplies periodic wrapping and sorted interval lookup; the interpolation shares return as scalars.
    """
    offsets = distances.to_array()
    wrapped = samples.to_array() % offsets[-1]
    indices = np.searchsorted(offsets, wrapped, side="right") - 1
    shares = (wrapped - offsets[indices]) / (offsets[indices + 1] - offsets[indices])
    mv = distances.context.multivector
    return indices, mv.scalar(shares[..., None]), mv.scalar(wrapped[..., None])


def arrow_along(curve: core.Point, direction: float, phase: float, arrow_length: float) -> core.Point:
    """An arrow along a closed curve, anchored at its rightmost crossing of its middle height and moved by
    a share of its perimeter in the current's direction."""
    mv = curve.context.multivector
    following = np.roll(np.arange(curve.shape[0]), -1)
    sides = curve[following] - curve                                        # [samples] Point-
    lengths = sides.squared().square_root()                                  # [samples] Scalar+
    distance = concatenate((mv.scalar([[0]]), lengths.cumsum(axis=0)))       # [samples + 1] Scalar+
    perimeter = distance[-1]
    heights = core.mv.y & curve                                              # [samples] Scalar+
    middle = (heights.min() + heights.max()) / 2
    crossing = np.flatnonzero((heights > middle) != (heights[following] > middle))
    shares = (middle - heights[crossing]) / (heights[following][crossing] - heights[crossing])
    middle_crossings = curve[crossing] + shares * sides[crossing]             # [crossings] Point-
    rightmost = (core.mv.x & middle_crossings).argmax()
    anchor = distance[crossing[rightmost]] + shares[rightmost] * lengths[crossing[rightmost]]
    half_length = (perimeter / 8).clip(0, arrow_length) / 2
    samples = anchor + direction * (phase * perimeter + half_length * np.array([-1, 1]))
    # Interpolate the two ends on their segments, retaining the curve's own points between them.
    side_idx, shares, samples = periodic_location(distance, samples)
    ends = curve[side_idx] + shares * sides[side_idx]                         # [2] Point-
    offsets = direction * (distance[:-1] - samples[0])
    offsets = where(offsets < 0, offsets + perimeter, offsets)
    inside = np.flatnonzero((offsets > 0) & (offsets < 2 * half_length))
    return concatenate((ends[:1], curve[inside[offsets[inside].argsort()]], ends[1:]))


def current_paths(ax: plt.Axes, mesh: Mesh, vertices: core.Point, stream: core.Scalar, gradients: core.Vector,
                  current: core.Current, levels: np.ndarray, arrow_length: float,
                  colour: tuple[float, float, float, float]) -> list[Artist]:
    """Closed current paths, the smoothed level lines at the levels `[levels]` of the streamfunction `Scalar+[vertices]`
    at the vertices where they are drawn, each with an arrow along it in the sense of the current `Current-[faces]`,
    the streamfunction's hats' gradients `[corners] Vector+[faces]` and the current taken in the material's frame; the
    artists drawn."""
    # The current runs with a loop where the streamfunction rises on the current's left, as it does on the loop's.
    stream_gradients = (gradients * (mesh.incidence[2, 0] * stream)).sum(axis=0)   # Vector+[faces]
    alignments = (stream_gradients & (core.quarter_turn >> current)).batch()
    senses = np.asarray(alignments > 0, dtype=int) - np.asarray(alignments < 0, dtype=int)  # [faces]
    paths, arrows = [], []
    for level, crossings, faces in level_loops(mesh, vertices, stream, levels):
        # The loop's sense against the current's, read on its longest segment.
        steps = crossings[np.roll(np.arange(len(faces)), -1)] - crossings    # [crossings] Point-
        direction = senses[faces[steps.squared().square_root().argmax()]]
        curve = smoothed(crossings, contour_passes)                         # [samples] Point-
        paths.append(Path(np.concatenate((planar(curve), planar(curve[:1]))), closed=True))
        # Arrows staggered between levels, keeping their phase as each path evolves.
        arrow = arrow_along(curve, direction, level * arrow_phase % 1, arrow_length)   # [arrow] Point-
        arrows.append(ax.add_artist(FancyArrowPatch(path=Path(planar(arrow)), arrowstyle="-|>", mutation_scale=7,
                                                    shrinkA=0, shrinkB=0, linewidth=0.8, color=colour, zorder=3)))
    loops = ax.add_collection(PathCollection(paths, facecolors="none", edgecolors=[colour], linewidths=0.8,
                                             linestyles="solid", zorder=3))
    return [loops] + arrows


def draw(mesh: Mesh, points: core.Point, areas: core.Scalar, response: core.Response, centre: core.Point,
         width: float, field: core.Bivector) -> plt.Figure:
    """The materials' solved currents on the mesh at its points, with magnitude on one colour scale, and the magnet's
    field, `field` at its centre, as its patch."""
    positions = planar(points.batch())
    triangles = Triangulation(positions[:, 0], positions[:, 1], mesh.faces)

    # Average the face currents at shared vertices solely to draw a continuous field.
    # The streamlines follow this interpolated solution; they are not prescribed paths.
    vertex_current = at_vertices(mesh, areas, response.current)            # [cases] Current-[vertices]
    widths = (stack((core.mv.x, core.mv.y)) & points.batch()[..., None]).abs().reshape(-1)
    extent = widths.max()
    sample_axis = extent * np.linspace(-1, 1, image_samples)
    samples = core.mv.x.dual() * sample_axis[None, :] + core.mv.y.dual() * sample_axis[:, None]
    corners = (mesh.incidence[2, 0] * points).batch()                          # [corners, faces] Point-
    gradients = corner_gradients(mesh, points).batch()                        # [corners, faces] Vector+
    # The hats of a point sum to one; their absolute values do so only inside the triangle.
    # Locate one image row at a time, so the hats for the whole image and every face are never held together.
    rows, masks = [], []
    for row in samples:
        candidates = (1 + (gradients[:, None] & (row[None, :, None] - corners[:, None]))).abs().sum(axis=0)
        nearest = candidates.argmin(axis=-1)
        rows.append(nearest)
        masks.append(candidates[np.arange(len(row)), nearest] <= 1 + 1e-9)
    holding, inside = np.stack(rows), np.stack(masks)
    sampled = sampling(mesh, points, samples[inside], holding[inside]) * vertex_current
    flows = np.full(vertex_current.shape + samples.shape + (2,), np.nan)     # [cases, rows, columns, 2]
    flows[:, inside] = planar(sampled.batch())
    magnitude = scalars(vertex_current.squared().square_root())            # [cases, vertices]
    current_scale = PowerNorm(current_gamma, vmin=0, vmax=magnitude.max())

    figure, panels = plt.subplots(1, len(flows), figsize=(3 * len(flows), 3.15),
                                  squeeze=False, layout="constrained")
    for ax, flow, speed in zip(panels[0], flows, magnitude):
        ax.tripcolor(triangles, speed, shading="gouraud", cmap=current_colours,
                     norm=current_scale, rasterized=True)
        ax.streamplot(sample_axis.to_array(), sample_axis.to_array(), flow[..., 0], flow[..., 1], color="#493c32",
                       density=1.0, linewidth=0.6, arrowsize=0.6)
        ax.add_collection(LineCollection(rim(mesh, points), colors="#746b60", linewidths=0.6))
        magnet(ax, centre, width, field)
        limits = (extent * np.array([-1, 1])).to_array()
        ax.set(xlim=limits, ylim=limits, aspect="equal")
        ax.set_axis_off()
    return figure


def draw_modes(mesh: Mesh, points: core.Point, streamfunctions: core.Scalar) -> plt.Figure:
    """Streamfunctions of current patterns `[cases, patterns] Scalar+[vertices]`, a row of patterns per
    material: current circulates along their contours, opposite ways around warm and cool peaks."""
    values = scalars(streamfunctions)                                      # [cases, patterns, vertices]
    positions = planar(points.batch())
    triangles = Triangulation(positions[:, 0], positions[:, 1], mesh.faces)
    extent = scalars(points.squared().square_root()).max()
    outline = rim(mesh, points)
    rows, columns = values.shape[:2]
    figure, panels = plt.subplots(rows, columns, figsize=(2 * columns, 2 * rows),
                                  squeeze=False, layout="constrained")
    for row, streams in zip(panels, values):
        for ax, stream in zip(row, streams):
            largest = np.abs(stream).max()
            ax.tripcolor(triangles, stream, shading="gouraud", cmap=mode_colours,
                         vmin=-largest, vmax=largest, rasterized=True)
            ax.tricontour(triangles, stream, levels=np.linspace(-largest, largest, mode_levels + 2)[1:-1],
                          colors="#493c32", linewidths=0.6, linestyles="solid")
            ax.add_collection(LineCollection(outline, colors="#746b60", linewidths=0.8))
            ax.set(xlim=(-extent, extent), ylim=(-extent, extent), aspect="equal")
            ax.set_axis_off()
    return figure


def animate(mesh: Mesh, points: core.Point, areas: core.Scalar, frames: Iterable[core.Motion], centre: core.Point,
            width: float, field: core.Bivector, fibres: tuple[core.Point, ...], shown: np.ndarray,
            room: float = current_room) -> Iterator[np.ndarray]:
    """Batched rotating materials and their currents, the currents' paths the contours of their streamfunctions, with
    `[curves, samples]` body-fixed guide families drawn on the panels that show them, `[cases, families]`. The colours
    share one magnitude scale across cases, and each panel keeps its contour values: both the first state's, `room`
    times over, widened where a later state's current exceeds them."""
    states = iter(frames)
    first = next(states)
    peak = room * scalars((first.orientation >> first.current).squared().square_root()).max()
    stream_top = room * np.abs(scalars(first.streamfunction)).max(axis=-1)     # [cases]
    yield from currents(mesh, points, areas, chain([first], states), centre, width, field, fibres, shown, peak,
                        stream_top)


def animate_induction(mesh: Mesh, points: core.Point, areas: core.Scalar, frames: Iterable[core.Motion],
                      centre: core.Point, width: float, field: core.Bivector, fibres: tuple[core.Point, ...],
                      shown: np.ndarray, peak: float = induction_current_peak) -> Iterator[np.ndarray]:
    """Rotating materials whose currents build up from none, drawn as `animate` draws them: the colours on a fixed
    magnitude scale up to `peak`, so that the currents fade in as they grow, and each panel's contour values widened
    with its current."""
    yield from currents(mesh, points, areas, frames, centre, width, field, fibres, shown, peak, np.zeros(len(shown)))


def currents(mesh: Mesh, points: core.Point, areas: core.Scalar, states: Iterable[core.Motion], centre: core.Point,
             width: float, field: core.Bivector, fibres: tuple[core.Point, ...], shown: np.ndarray, peak: float,
             stream_top: np.ndarray) -> Iterator[np.ndarray]:
    """The currents' frames, from a colour scale up to `peak` and contour values up to `stream_top` per case, each
    widened where a state's current exceeds it."""
    # Separate contour ranges keep the weak material cases legible; colours share one magnitude scale.
    fractions = (np.arange(-current_levels, current_levels) + 0.5) / current_levels
    extent = scalars(points.squared().square_root()).max()
    # The streamfunction's hats on the material's faces, which turn with them.
    gradients = corner_gradients(mesh, points)                                # [corners] Vector+[faces]
    guides = concatenate(fibres)                                              # [curves, samples] Point-

    figure = plt.figure(figsize=(3.3 * len(shown), 3.6), dpi=animation_dpi)
    panels = [disc_panel(figure, i, len(shown), extent, centre, width, field, colours)
              for i, colours in enumerate(fibre_colours(fibres, shown))]
    for state in states:
        # The current belongs to material faces; turning it into the laboratory frame keeps its strength.
        strength = state.current.squared().square_root()                      # [cases] Scalar+[faces]
        magnitude = scalars(strength)                                         # [cases, faces]
        stream = scalars(state.streamfunction)                                # [cases, vertices]
        # The scales widened where this state's current exceeds them.
        peak, stream_top = max(peak, magnitude.max()), np.maximum(stream_top, np.abs(stream).max(axis=-1))
        levels = stream_top[:, None] * fractions                              # [cases, levels]
        alpha = np.divide(magnitude.max(axis=-1), peak, out=np.zeros(len(magnitude)),
                          where=peak > 0) ** streamline_gamma
        current_scale = PowerNorm(current_gamma, vmin=0, vmax=peak)
        speed = scalars(at_vertices(mesh, areas, strength))                   # [cases, vertices]
        positions = planar(state.vertices.batch())
        threads_at = planar(state.orientation[:, None, None] >> guides)
        marks = rotation_marks(state.orientation, extent)                     # [cases, 2]
        drawn = []
        for i, (ax, threads, marker, outline) in enumerate(panels):
            outline.set_segments(rim(mesh, state.vertices[i]))
            threads.set_segments(threads_at[i])
            marker.set_offsets(marks[i:i + 1])
            triangles = Triangulation(positions[i, :, 0], positions[i, :, 1], mesh.faces)
            drawn.append(ax.tripcolor(triangles, speed[i], shading="gouraud",
                                      cmap=current_colours, norm=current_scale, rasterized=True, zorder=1))
            colour = (0.29, 0.24, 0.20, alpha[i])
            drawn += current_paths(ax, mesh, state.vertices[i], state.streamfunction[i], gradients, state.current[i],
                                   levels[i, levels[i] != 0], current_arrow_length * extent, colour)
        yield capture(figure)
        for artist in drawn:
            artist.remove()
    plt.close(figure)


def animate_heat(mesh: Mesh, points: core.Point, areas: core.Scalar, frames: Iterable[core.Motion], centre: core.Point,
                 width: float, field: core.Bivector, fibres: tuple[core.Point, ...], shown: np.ndarray,
                 peak: float = braking_heat_peak) -> Iterator[np.ndarray]:
    """Accumulated heat per area, carried by the moving material faces on one fixed scale up to `peak`."""
    heat_scale = PowerNorm(heat_gamma, vmin=0, vmax=peak)
    extent = scalars(points.squared().square_root()).max()
    guides = concatenate(fibres)                                              # [curves, samples] Point-

    figure = plt.figure(figsize=(3.3 * len(shown), 3.6), dpi=animation_dpi)
    panels = [disc_panel(figure, i, len(shown), extent, centre, width, field, colours)
              for i, colours in enumerate(fibre_colours(fibres, shown))]
    for state in frames:
        # Face heat is energy. Dividing by each material face's area gives the density to colour.
        densities = scalars(at_vertices(mesh, areas, state.heat / areas))     # [cases, vertices]
        positions = planar(state.vertices.batch())
        threads_at = planar(state.orientation[:, None, None] >> guides)
        marks = rotation_marks(state.orientation, extent)                     # [cases, 2]
        surfaces = []
        for i, (ax, threads, marker, outline) in enumerate(panels):
            # The display mesh moves with the material; vertex colours interpolate face heat.
            triangles = Triangulation(positions[i, :, 0], positions[i, :, 1], mesh.faces)
            surfaces.append(ax.tripcolor(triangles, densities[i], shading="gouraud",
                                          cmap=heat_colours, norm=heat_scale, rasterized=True, zorder=1))
            outline.set_segments(rim(mesh, state.vertices[i]))
            threads.set_segments(threads_at[i])
            marker.set_offsets(marks[i:i + 1])
        yield capture(figure)
        for surface in surfaces:
            surface.remove()
    plt.close(figure)


def animate_induction_heat(mesh: Mesh, points: core.Point, areas: core.Scalar, frames: Iterable[core.Motion],
                           centre: core.Point, width: float, field: core.Bivector, fibres: tuple[core.Point, ...],
                           shown: np.ndarray, peak: float = induction_heat_peak) -> Iterator[np.ndarray]:
    """The heat of the discs spun up from rest, drawn as `animate_heat` draws it, on its fixed scale up to `peak`."""
    yield from animate_heat(mesh, points, areas, frames, centre, width, field, fibres, shown, peak)
