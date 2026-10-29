"""Current paths in a conducting disc beneath a local magnetic field."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.path import Path
from matplotlib.tri import LinearTriInterpolator, Triangulation, TriContourSet

from numga import concatenate, stack
from numga.extensor import Extensor
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


def magnet(ax: plt.Axes, field_centre: np.ndarray, width: float, field: core.Bivector) -> None:
    """The magnet's field, a plane of space, as a patch in that plane: its width dashed, and an arc around it with an
    arrowhead for the sense the field turns in, read by pairing the field with the patch of the disc's plane."""
    ax.add_patch(Circle(field_centre, width, facecolor="none", edgecolor=magnet_colour,
                        linewidth=0.75, linestyle=(0, (3, 3)), zorder=4))
    sense = np.sign((field & core.mv.xy.dual()).to_array())
    angles = field_arc_start + sense * np.linspace(0, field_arc_span, n_field_arc)
    arc = field_centre + field_arc_reach * width * np.stack((np.cos(angles), np.sin(angles)), axis=-1)
    ax.add_patch(FancyArrowPatch(path=Path(arc), arrowstyle="-|>", mutation_scale=8, shrinkA=0, shrinkB=0,
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


def disc_panel(figure: plt.Figure, index: int, count: int, extent: float, field_centre: np.ndarray,
               width: float, field: core.Bivector,
               colours: np.ndarray) -> tuple[plt.Axes, LineCollection, PathCollection, LineCollection]:
    """One panel of a row of rotating discs: fibre guides, rim knob, rim outline and the magnet."""
    ax = figure.add_axes(((index + 0.035) / count, 0.03, 0.93 / count, 0.86))
    threads = LineCollection([], colors=colours, linewidths=fibre_width, zorder=2)
    ax.add_collection(threads)
    marker = rotation_marker(ax)
    outline = LineCollection([], colors="#746b60", linewidths=0.8, zorder=3)
    ax.add_collection(outline)
    magnet(ax, field_centre, width, field)
    ax.set(xlim=(-extent, extent), ylim=(-extent, extent), aspect="equal")
    ax.set_axis_off()
    return ax, threads, marker, outline


def smooth_contour(curve: np.ndarray, passes: int) -> np.ndarray:
    """Chaikin corner cutting on a closed `[points, 2]` display curve."""
    points = curve[:-1]
    for _ in range(passes):
        following = np.roll(points, -1, axis=0)
        points = np.stack((0.75 * points + 0.25 * following,
                           0.25 * points + 0.75 * following), axis=1).reshape(-1, 2)
    return np.concatenate((points, points[:1]))


def current_contours(ax: plt.Axes, triangles: Triangulation, streamfunction: np.ndarray,
                     current: np.ndarray, levels: np.ndarray, arrow_length: float,
                     colour: tuple[float, float, float, float]) -> TriContourSet:
    """Closed current paths at prescribed levels, with geometrically anchored direction arrows."""
    contours = ax.tricontour(triangles, streamfunction, levels=levels, colors=[colour],
                             linewidths=0.8, linestyles="solid", zorder=3)
    locate = triangles.get_trifinder()
    paths = []
    for level, pieces in enumerate(contours.allsegs):
        display_paths = []
        for raw in (curve for curve in pieces if len(curve) > 2):
            # Read the solved current's direction on the unsmoothed contour's longest segment.
            segments = raw[1:] - raw[:-1]
            longest = np.linalg.norm(segments, axis=-1).argmax()
            midpoint = (raw[longest] + raw[longest + 1]) / 2
            direction = np.sign(segments[longest] @ current[locate(*midpoint)])
            curve = smooth_contour(raw, contour_passes)
            display_paths.append(Path(curve, closed=True))
            starts, ends = curve[:-1], curve[1:]
            lengths = np.linalg.norm(ends - starts, axis=-1)
            distance = np.concatenate(([0], np.cumsum(lengths)))
            perimeter = distance[-1]
            # Anchor at a horizontal crossing, independent of the contour tracer's start vertex.
            height = (curve[:, 1].min() + curve[:, 1].max()) / 2
            crossings = np.flatnonzero((starts[:, 1] > height) != (ends[:, 1] > height))
            fractions = (height - starts[crossings, 1]) / (ends[crossings, 1] - starts[crossings, 1])
            horizontal = starts[crossings, 0] + fractions * (ends[crossings, 0] - starts[crossings, 0])
            rightmost = horizontal.argmax()
            anchor = distance[crossings[rightmost]] + fractions[rightmost] * lengths[crossings[rightmost]]
            half_length = min(arrow_length, perimeter / 8) / 2
            # Stagger arrows between levels, retaining their phase as each path evolves.
            samples = (anchor + direction * ((level * arrow_phase % 1) * perimeter
                       + np.array([-half_length, half_length]))) % perimeter
            points = np.stack((np.interp(samples, distance, curve[:, 0]),
                               np.interp(samples, distance, curve[:, 1])), axis=-1)
            # Follow the smoothed curve itself, so arrows stay on the displayed current path.
            offsets = direction * (distance[:-1] - samples[0]) % perimeter
            inside = np.flatnonzero((offsets > 0) & (offsets < 2 * half_length))
            inside = inside[np.argsort(offsets[inside])]
            points = np.concatenate((points[:1], curve[inside], points[1:]))
            ax.add_artist(FancyArrowPatch(path=Path(points), arrowstyle="-|>", mutation_scale=7,
                                         shrinkA=0, shrinkB=0, linewidth=0.8, color=colour, zorder=3))
        paths.append(Path.make_compound_path(*display_paths))
    contours.set_paths(paths)
    return contours


def draw(mesh: Mesh, points: core.Point, areas: core.Scalar, response: core.Response, centre: core.Point,
         width: float, field: core.Bivector) -> plt.Figure:
    """The materials' solved currents on the mesh at its points, with magnitude on one colour scale, and the magnet's
    field, `field` at its centre, as its patch."""
    positions = planar(points.batch())
    field_centre = planar(centre)
    triangles = Triangulation(positions[:, 0], positions[:, 1], mesh.faces)

    # Average the face currents at shared vertices solely to draw a continuous field.
    # The streamlines follow this interpolated solution; they are not prescribed paths.
    vertex_current = planar(at_vertices(mesh, areas, response.current).batch())   # [cases, vertices, 2]
    extent = np.abs(positions).max()
    sample_axis = np.linspace(-extent, extent, image_samples)
    horizontal, vertical = np.meshgrid(sample_axis, sample_axis)
    magnitude = np.linalg.norm(vertex_current, axis=-1)
    current_scale = PowerNorm(current_gamma, vmin=0, vmax=magnitude.max())
    boundary = mesh.edges[mesh.boundary_edges]

    figure, panels = plt.subplots(1, len(vertex_current), figsize=(3 * len(vertex_current), 3.15),
                                  squeeze=False, layout="constrained")
    for ax, vectors, speed in zip(panels[0], vertex_current, magnitude):
        along = LinearTriInterpolator(triangles, vectors[:, 0])(horizontal, vertical)
        across = LinearTriInterpolator(triangles, vectors[:, 1])(horizontal, vertical)
        ax.tripcolor(triangles, speed, shading="gouraud", cmap=current_colours,
                     norm=current_scale, rasterized=True)
        ax.streamplot(sample_axis, sample_axis, along, across, color="#493c32",
                       density=1.0, linewidth=0.6, arrowsize=0.6)
        ax.add_collection(LineCollection(positions[boundary], colors="#746b60", linewidths=0.6))
        magnet(ax, field_centre, width, field)
        ax.set(xlim=(-extent, extent), ylim=(-extent, extent), aspect="equal")
        ax.set_axis_off()
    return figure


def draw_modes(mesh: Mesh, points: core.Point, streamfunctions: core.Scalar) -> plt.Figure:
    """Streamfunctions of current patterns `[cases, patterns] Scalar+[vertices]`, a row of patterns per
    material: current circulates along their contours, opposite ways around warm and cool peaks."""
    values = scalars(streamfunctions)                                      # [cases, patterns, vertices]
    positions = planar(points.batch())
    triangles = Triangulation(positions[:, 0], positions[:, 1], mesh.faces)
    extent = np.linalg.norm(positions, axis=-1).max()
    boundary = mesh.edges[mesh.boundary_edges]
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
            ax.add_collection(LineCollection(positions[boundary], colors="#746b60", linewidths=0.8))
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
    reference = planar(points.batch())
    extent = np.linalg.norm(reference, axis=-1).max()
    field_centre = planar(centre)
    boundary = mesh.edges[mesh.boundary_edges]
    guides = concatenate(fibres)                                              # [curves, samples] Point-

    figure = plt.figure(figsize=(3.3 * len(shown), 3.6), dpi=animation_dpi)
    panels = [disc_panel(figure, i, len(shown), extent, field_centre, width, field, colours)
              for i, colours in enumerate(fibre_colours(fibres, shown))]
    fixed_patches = len(panels[0][0].patches)
    for state in states:
        # The current belongs to material faces; turn its direction into the laboratory frame.
        turned_current = state.orientation >> state.current                   # [cases] Current-[faces]
        magnitude = scalars(turned_current.squared().square_root())           # [cases, faces]
        vectors = planar(turned_current.batch())                              # [cases, faces, 2]
        stream = scalars(state.streamfunction)                                # [cases, vertices]
        # The scales widened where this state's current exceeds them.
        peak, stream_top = max(peak, magnitude.max()), np.maximum(stream_top, np.abs(stream).max(axis=-1))
        levels = stream_top[:, None] * fractions                              # [cases, levels]
        alpha = np.divide(magnitude.max(axis=-1), peak, out=np.zeros(len(magnitude)),
                          where=peak > 0) ** streamline_gamma
        current_scale = PowerNorm(current_gamma, vmin=0, vmax=peak)
        speed = scalars(at_vertices(mesh, areas, turned_current.squared().square_root()))   # [cases, vertices]
        positions = planar(state.vertices.batch())
        threads_at = planar(state.orientation[:, None, None] >> guides)
        marks = rotation_marks(state.orientation, extent)                     # [cases, 2]
        paths, surfaces = [], []
        for i, (ax, threads, marker, outline) in enumerate(panels):
            outline.set_segments(positions[i, boundary])
            threads.set_segments(threads_at[i])
            marker.set_offsets(marks[i:i + 1])
            triangles = Triangulation(positions[i, :, 0], positions[i, :, 1], mesh.faces)
            surfaces.append(ax.tripcolor(triangles, speed[i], shading="gouraud",
                                          cmap=current_colours, norm=current_scale, rasterized=True, zorder=1))
            colour = (0.29, 0.24, 0.20, alpha[i])
            paths.append(current_contours(ax, triangles, stream[i], vectors[i], levels[i, levels[i] != 0],
                                           current_arrow_length * extent, colour))
        yield capture(figure)
        for surface in surfaces:
            surface.remove()
        for path, (ax, threads, marker, outline) in zip(paths, panels):
            path.remove()
            for arrow in tuple(ax.patches)[fixed_patches:]:
                arrow.remove()
    plt.close(figure)


def animate_heat(mesh: Mesh, points: core.Point, areas: core.Scalar, frames: Iterable[core.Motion], centre: core.Point,
                 width: float, field: core.Bivector, fibres: tuple[core.Point, ...], shown: np.ndarray,
                 peak: float = braking_heat_peak) -> Iterator[np.ndarray]:
    """Accumulated heat per area, carried by the moving material faces on one fixed scale up to `peak`."""
    heat_scale = PowerNorm(heat_gamma, vmin=0, vmax=peak)
    reference = planar(points.batch())
    extent = np.linalg.norm(reference, axis=-1).max()
    field_centre = planar(centre)
    boundary = mesh.edges[mesh.boundary_edges]
    guides = concatenate(fibres)                                              # [curves, samples] Point-

    figure = plt.figure(figsize=(3.3 * len(shown), 3.6), dpi=animation_dpi)
    panels = [disc_panel(figure, i, len(shown), extent, field_centre, width, field, colours)
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
            outline.set_segments(positions[i, boundary])
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
