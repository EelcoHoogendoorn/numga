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

from numga import concatenate, stack
from numga.extensor import Extensor
from examples.util.animation import capture
from examples.util.wgsl import contours, pinned, solids
from examples.util.wgsl.flow import axes_view
from examples.electromagnetism.eddy_brake import core

Mesh, as_scalar = core.Mesh, core.as_scalar
# The observer's slice of space, which carries the disc's events into the renderer's space.
space_slice = core.mv.t ^ (core.Point & core.time)                              # [] Point- <- Point-

animation_dpi = 90
fibre_reach = 0.94
fibre_colour = (0.16, 0.43, 0.56, 0.8)
fibre_width = 1.35
current_gamma = 0.35
streamline_gamma = 0.25
current_levels = 10
# The ink of the current paths and of the current patterns' contours: colour and opacity.
path_ink = np.array([0.29, 0.24, 0.20, 1.0])
mode_ink = np.array([0.29, 0.24, 0.20, 1.0])
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


def draw(mesh: Mesh, points: core.Point, areas: core.Scalar, response: core.Response, streamfunction: core.Scalar,
         centre: core.Point, width: float, field: core.Bivector) -> plt.Figure:
    """The materials' solved currents on the mesh at its points, their strength on one colour scale and their paths
    the contours of their streamfunctions `[cases] Scalar+[vertices]`, and the magnet's field, `field` at its centre, as
    its patch."""
    # Separate contour steps keep the weak material cases legible; colours share one magnitude scale.
    streams = scalars(streamfunction)                                       # [cases, vertices]
    spacings = np.abs(streams).max(axis=-1) / current_levels                # [cases]
    magnitude = scalars(at_vertices(mesh, areas, response.current).squared().square_root())   # [cases, vertices]
    current_scale = PowerNorm(current_gamma, vmin=0, vmax=magnitude.max())
    extent = scalars(points.squared().square_root()).max()
    places = solids.origin + solids.context.include_from_algebra(space_slice(points).batch())   # [vertices] Point-

    figure, panels = plt.subplots(1, len(magnitude), figsize=(3 * len(magnitude), 3.15),
                                  squeeze=False, layout="constrained")
    for ax, stream, spacing, speed in zip(panels[0], streams, spacings, magnitude):
        ax.add_collection(LineCollection(rim(mesh, points), colors="#746b60", linewidths=0.6))
        magnet(ax, centre, width, field)
        ax.set(xlim=(-extent, extent), ylim=(-extent, extent), aspect="equal")
        ax.set_axis_off()
        colours = current_colours(current_scale(speed))[:, :3]
        # The paths halfway between the multiples of the step, so that none runs along the rim, where the
        # streamfunction vanishes.
        paths = contours.Lines(spacing, spacing / 2, path_ink, 1.0)
        contours.overlay(ax, places, mesh, stream, colours, lines=(paths,)).set_zorder(1)
    return figure


def draw_modes(mesh: Mesh, points: core.Point, streamfunctions: core.Scalar) -> plt.Figure:
    """Streamfunctions of current patterns `[cases, patterns] Scalar+[vertices]`, a row of patterns per
    material: current circulates along their contours, opposite ways around warm and cool peaks."""
    values = scalars(streamfunctions)                                      # [cases, patterns, vertices]
    extent = scalars(points.squared().square_root()).max()
    outline = rim(mesh, points)
    places = solids.origin + solids.context.include_from_algebra(space_slice(points).batch())   # [vertices] Point-
    rows, columns = values.shape[:2]
    figure, panels = plt.subplots(rows, columns, figsize=(2 * columns, 2 * rows),
                                  squeeze=False, layout="constrained")
    for row, streams in zip(panels, values):
        for ax, stream in zip(row, streams):
            largest = np.abs(stream).max()
            ax.add_collection(LineCollection(outline, colors="#746b60", linewidths=0.8))
            ax.set(xlim=(-extent, extent), ylim=(-extent, extent), aspect="equal")
            ax.set_axis_off()
            # Levels at equal steps between the extremes, one of them through zero.
            colours = mode_colours((stream / largest + 1) / 2)[:, :3]
            levels = contours.Lines(2 * largest / (mode_levels + 1), 0.0, mode_ink, 1.0)
            contours.overlay(ax, places, mesh, stream, colours, lines=(levels,)).set_zorder(1)
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
    extent = scalars(points.squared().square_root()).max()
    guides = concatenate(fibres)                                              # [curves, samples] Point-

    figure = plt.figure(figsize=(3.3 * len(shown), 3.6), dpi=animation_dpi)
    panels = [disc_panel(figure, i, len(shown), extent, centre, width, field, colours)
              for i, colours in enumerate(fibre_colours(fibres, shown))]
    views = [axes_view(ax) for ax, _, _, _ in panels]
    textures = [pinned(ax, np.zeros(view.shape + (4,)), (*ax.get_xlim(), *ax.get_ylim()))
                for (ax, _, _, _), view in zip(panels, views)]
    for texture in textures:
        texture.set_zorder(1)
    for state in states:
        # The current belongs to material faces; turning it into the laboratory frame keeps its strength.
        strength = state.current.squared().square_root()                      # [cases] Scalar+[faces]
        magnitude = scalars(strength)                                         # [cases, faces]
        stream = scalars(state.streamfunction)                                # [cases, vertices]
        # The scales widened where this state's current exceeds them.
        peak, stream_top = max(peak, magnitude.max()), np.maximum(stream_top, np.abs(stream).max(axis=-1))
        # Separate contour steps keep the weak material cases legible; a disc still without current has no levels.
        spacings = np.where(stream_top > 0, stream_top, 1.0) / current_levels   # [cases]
        alpha = np.divide(magnitude.max(axis=-1), peak, out=np.zeros(len(magnitude)),
                          where=peak > 0) ** streamline_gamma
        current_scale = PowerNorm(current_gamma, vmin=0, vmax=peak)
        speed = scalars(at_vertices(mesh, areas, strength))                   # [cases, vertices]
        places = solids.origin + solids.context.include_from_algebra(space_slice(state.vertices).batch())   # [cases, vertices] Point-
        threads_at = planar(state.orientation[:, None, None] >> guides)
        marks = rotation_marks(state.orientation, extent)                     # [cases, 2]
        for i, ((ax, threads, marker, outline), view, texture) in enumerate(zip(panels, views, textures)):
            outline.set_segments(rim(mesh, state.vertices[i]))
            threads.set_segments(threads_at[i])
            marker.set_offsets(marks[i:i + 1])
            colours = current_colours(current_scale(speed[i]))[:, :3]
            ink = np.append(path_ink[:3], alpha[i])
            paths = contours.Lines(spacings[i], spacings[i] / 2, ink, 1.0)
            texture.set_data(contours.picture(places[i], mesh, stream[i], colours, view, lines=(paths,)))
        yield capture(figure)
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
    views = [axes_view(ax) for ax, _, _, _ in panels]
    textures = [pinned(ax, np.zeros(view.shape + (4,)), (*ax.get_xlim(), *ax.get_ylim()))
                for (ax, _, _, _), view in zip(panels, views)]
    for texture in textures:
        texture.set_zorder(1)
    for state in frames:
        # Face heat is energy. Dividing by each material face's area gives the density to colour.
        densities = scalars(at_vertices(mesh, areas, state.heat / areas))     # [cases, vertices]
        places = solids.origin + solids.context.include_from_algebra(space_slice(state.vertices).batch())   # [cases, vertices] Point-
        threads_at = planar(state.orientation[:, None, None] >> guides)
        marks = rotation_marks(state.orientation, extent)                     # [cases, 2]
        for i, ((ax, threads, marker, outline), view, texture) in enumerate(zip(panels, views, textures)):
            # The display mesh moves with the material; its colours interpolate face heat, with no level lines.
            colours = heat_colours(heat_scale(densities[i]))[:, :3]
            texture.set_data(contours.picture(places[i], mesh, densities[i], colours, view))
            outline.set_segments(rim(mesh, state.vertices[i]))
            threads.set_segments(threads_at[i])
            marker.set_offsets(marks[i:i + 1])
        yield capture(figure)
    plt.close(figure)


def animate_induction_heat(mesh: Mesh, points: core.Point, areas: core.Scalar, frames: Iterable[core.Motion],
                           centre: core.Point, width: float, field: core.Bivector, fibres: tuple[core.Point, ...],
                           shown: np.ndarray, peak: float = induction_heat_peak) -> Iterator[np.ndarray]:
    """The heat of the discs spun up from rest, drawn as `animate_heat` draws it, on its fixed scale up to `peak`."""
    yield from animate_heat(mesh, points, areas, frames, centre, width, field, fibres, shown, peak)
