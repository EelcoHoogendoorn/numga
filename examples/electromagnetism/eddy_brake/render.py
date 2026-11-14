"""Current streamfunction contours, rotating fibre guides, and accumulated Joule heat density on a
triangulated conducting disc beneath a localized magnetic field."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, PowerNorm

from numga import Algebra, NumpyContext, concatenate
from numga.algebras import STA2D
from numga.extensor import Extensor
from numga.render import Camera, contours, drawing, simplices
from numga.topology import Topology
from examples.util.animation import capture
from examples.util.panels import offset_panel, overlay

ga = STA2D
Scalar = ga.gatype.scalar()                                                  # Scalar+
Point = ga.gatype.antivector()                                               # Antivector-
Bivector = ga.gatype.bivector()                                              # Bivector+
Rotor = ga.gatype.rotor()                                                    # Rotor+
mv = NumpyContext(ga).multivector
Current = ga.gatype.vector().derive.within(Algebra("x+y+")).derive.dual      # Antivector-

animation_dpi = 90
fibre_reach = 0.94
fibre_colour, fibre_opacity = np.array([0.16, 0.43, 0.56]), 0.8
fibre_width = 1.7
rim_colour, rim_width = np.array([0x74, 0x6b, 0x60]) / 255, 1.0
current_gamma = 0.35
streamline_gamma = 0.25
n_current_levels = 10
path_ink = np.array([0.29, 0.24, 0.20, 1.0])
mode_ink = np.array([0.29, 0.24, 0.20, 1.0])
field_arc_reach = 1.3
field_arc_start, field_arc_span = np.radians(-60.0), np.radians(240.0)
n_field_arc = 24
n_magnet_rim = 64
magnet_colour = np.array([0x33, 0x41, 0x55]) / 255
magnet_width, magnet_dashes, magnet_chevron = 1.0, (4.0, 4.0), 6.0
knob_colour, knob_ring_colour = np.array([1.0, 0.0, 0.784]), np.ones(3)
knob_radius, knob_ring_radius = 6.5, 8.4
heat_gamma = 0.3
current_colours = LinearSegmentedColormap.from_list("current", ("#f8f5ed", "#edb15f", "#c95832"))
heat_colours = LinearSegmentedColormap.from_list("heat", ("#fbf9f4", "#e58b3a", "#8f2d1b"))
mode_colours = LinearSegmentedColormap.from_list("mode", ("#2b6c8f", "#f8f5ed", "#c95832"))
n_mode_levels = 9
current_room = 1.0
induction_current_peak = 8.55e5
braking_heat_peak = 1.61
induction_heat_peak = 1.94e4


# --- plumbing -------------------------------------------------------------------------
def scalars(values: Scalar) -> np.ndarray:
    """Plain NumPy values read out from a scalar batch or field (`values`) feed colormapping and contouring."""
    return values.batch().to_array()


def magnet(centre: Point, width: float, field: Bivector, camera: Camera) -> list[np.ndarray]:
    """A dashed width circle and a directional arc ending in a chevron mark the magnetic patch,
    with turning sense matching `field & xy.dual()`."""
    rim_angles = np.linspace(0, 2 * np.pi, n_magnet_rim, endpoint=False)
    rim_turns = (mv.xy * (-rim_angles / 2)).exp()                       # [n_magnet_rim] Rotor+
    rim_offsets = rim_turns >> (mv.x.dual() * width)                    # [n_magnet_rim] Point-
    outline = centre + rim_offsets                                           # [n_magnet_rim] Point-
    handedness = field & mv.xy.dual()                                   # [] Scalar+
    sense = np.asarray(handedness > 0, dtype=int) - np.asarray(handedness < 0, dtype=int)
    arc_angles = field_arc_start + sense * np.linspace(0, field_arc_span, n_field_arc)
    arc_turns = (mv.xy * (-arc_angles / 2)).exp()                       # [n_field_arc] Rotor+
    arc_offsets = arc_turns >> (mv.x.dual() * (field_arc_reach * width))  # [n_field_arc] Point-
    arc = centre + arc_offsets                                               # [n_field_arc] Point-
    return [simplices.strokes(outline, Topology.loops([n_magnet_rim]), magnet_colour, magnet_width, camera,
                              dashes=magnet_dashes, cap=simplices.Cap.BUTT),
            simplices.strokes(arc, Topology.chains([n_field_arc]), magnet_colour, magnet_width, camera,
                              chevrons=np.arange(n_field_arc - 1) == n_field_arc - 2, chevron=magnet_chevron)]


def parallel_fibres(radius: float, n_curves: int, n_samples: int) -> Point:
    """Parallel chords (`[n_curves, n_samples] Point-`) mark straight body-fixed fibre directions."""
    offsets = np.linspace(-fibre_reach, fibre_reach, n_curves)
    fractions = np.linspace(-1, 1, n_samples)
    half_lengths = radius * np.sqrt(1 - offsets ** 2)
    return (mv.x.dual() * half_lengths[:, None] * fractions[None, :]
            + mv.y.dual() * radius * offsets[:, None])                  # [n_curves, n_samples] Point-


def radial_fibres(inner_radius: float, outer_radius: float, n_curves: int, n_samples: int) -> Point:
    """Radial rays (`[2 * n_curves, n_samples] Point-`) mark radial fibres across the annulus."""
    n_rays = 2 * n_curves
    angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    distances = np.linspace(inner_radius, outer_radius, n_samples)
    turns = (mv.xy * (-angles / 2)).exp()                               # [n_rays] Rotor+
    directions = turns >> mv.x.dual()                                   # [n_rays] Point-
    return directions[:, None] * distances[None, :]                          # [n_rays, n_samples] Point-


def circular_fibres(inner_radius: float, outer_radius: float, n_curves: int, n_samples: int) -> Point:
    """Concentric rings (`[n_curves, n_samples] Point-`) mark circumferential fibres across the annulus."""
    radii = np.linspace(inner_radius, outer_radius, n_curves)
    angles = np.linspace(0, 2 * np.pi, n_samples)
    turns = (mv.xy * (-angles / 2)).exp()                               # [n_samples] Rotor+
    directions = turns >> mv.x.dual()                                   # [n_samples] Point-
    return directions[None, :] * radii[:, None]                              # [n_curves, n_samples] Point-


def rotation_marks(orientations: Rotor, radius: float) -> Point:
    """Body-fixed rim marker positions (`[...] Point-`) track each rotor in `orientations`."""
    return orientations >> (mv.x.dual() * (fibre_reach * radius))       # [...] Point-


def rotation_marker(mark: Point, camera: Camera) -> list[np.ndarray]:
    """A fluorescent rim knob inside a white ring marks the rotated rim point `mark` (`[1] Point-`)."""
    return [simplices.markers(mark, knob_ring_colour, knob_ring_radius, camera),
            simplices.markers(mark, knob_colour, knob_radius, camera)]


def vertex_average(topology: Topology, areas: Scalar, face_field: Extensor) -> Extensor:
    """Area-weighted corner averaging maps a triangle field `[...] Scalar+[F]` or `[...] Current-[F]`
    to vertices (`[V]`)."""
    third = topology.incidence[2, 0].mean(axis=0)                            # Scalar+[F <- V]
    return (third.adjugate() * (areas * face_field)) / (third.adjugate() * areas)


def fibre_strokes(threads: Point, shown: np.ndarray, camera: Camera) -> np.ndarray:
    """Polyline strokes trace the selected guide curves in `threads` (`[..., n_samples] Point-`)."""
    visible = threads[np.flatnonzero(shown)]                                 # [..., n_samples] Point-
    n_shown, n_samples = visible.shape
    return simplices.strokes(visible.reshape(-1), Topology.chains([n_samples] * n_shown), fibre_colour, fibre_width,
                             camera, opacity=fibre_opacity)


def disc_panel(figure: plt.Figure, index: int, n_panels: int, extent: float) -> plt.Axes:
    """Panel `index` of `n_panels` frames one rotating disc in a horizontal row."""
    ax = figure.add_axes(((index + 0.035) / n_panels, 0.03, 0.93 / n_panels, 0.86))
    ax.set(xlim=(-extent, extent), ylim=(-extent, extent), aspect="equal")
    ax.set_axis_off()
    return ax


def steady_figure(topology: Topology, points: Point, areas: Scalar,
                  current: Current, streamfunction: Scalar,
                  centre: Point, width: float, field: Bivector) -> plt.Figure:
    """Steady sheet currents (`[...] Current-[F]`) and streamfunction contours (`[...] Scalar+[V]`)
    share one magnitude colour scale alongside the magnetic patch."""
    streams = scalars(streamfunction)                                        # [..., V]
    spacings = np.abs(streams).max(axis=-1) / n_current_levels               # [...]
    magnitude = scalars(vertex_average(topology, areas, current).antinorm())  # [..., V]
    current_scale = PowerNorm(current_gamma, vmin=0, vmax=magnitude.max())
    extent = float(points.antinorm().batch().max().to_array())
    places = points.batch()                                                  # [V] Point-

    outline = topology.border
    n_panels = len(magnitude)
    figure, panels = plt.subplots(1, n_panels, figsize=(3 * n_panels, 3.15),
                                  squeeze=False, layout="constrained")
    for ax, stream, spacing, speed in zip(panels[0], streams, spacings, magnitude):
        ax.set(xlim=(-extent, extent), ylim=(-extent, extent), aspect="equal")
        ax.set_axis_off()
        camera = offset_panel(ax, mv.x, mv.y)
        colours = current_colours(current_scale(speed))[:, :3]
        paths = contours.Lines(spacing, spacing / 2, path_ink, 1.0)
        overlay(ax, [contours.picture(places, topology, stream, colours, camera, lines=(paths,)),
                     simplices.strokes(places, outline, rim_colour, rim_width, camera),
                     *magnet(centre, width, field, camera)])
    return figure


def modes_figure(topology: Topology, points: Point, streamfunctions: Scalar) -> plt.Figure:
    """Streamfunction contours (`[..., n_shown_modes] Scalar+[V]`) arrange one row per material
    with opposite circulation senses around warm and cool peaks."""
    values = scalars(streamfunctions)                                        # [..., V]
    extent = float(points.antinorm().batch().max().to_array())
    outline = topology.border
    places = points.batch()                                                  # [V] Point-
    n_rows, n_columns = values.shape[:2]
    figure, panels = plt.subplots(n_rows, n_columns, figsize=(2 * n_columns, 2 * n_rows),
                                  squeeze=False, layout="constrained")
    for row, streams in zip(panels, values):
        for ax, stream in zip(row, streams):
            largest = np.abs(stream).max()
            ax.set(xlim=(-extent, extent), ylim=(-extent, extent), aspect="equal")
            ax.set_axis_off()
            camera = offset_panel(ax, mv.x, mv.y)
            colours = mode_colours((stream / largest + 1) / 2)[:, :3]
            levels = contours.Lines(2 * largest / (n_mode_levels + 1), 0.0, mode_ink, 1.0)
            overlay(ax, [contours.picture(places, topology, stream, colours, camera, lines=(levels,)),
                         simplices.strokes(places, outline, rim_colour, rim_width, camera)])
    return figure


def animate_currents(topology: Topology, points: Point, areas: Scalar,
                     centre: Point, width: float, field: Bivector,
                     fibres: tuple[Point, ...], shown: np.ndarray,
                     states: Iterable[tuple]) -> Iterator[np.ndarray]:
    """Yields braking frames with current magnitudes and streamfunction contours scaled to the
    initial state and widened whenever a later state exceeds it."""
    iterator = iter(states)
    first = next(iterator)
    orientation, _, _, _, current, streamfunction = first[:6]
    peak = current_room * scalars((orientation >> current).antinorm()).max()
    stream_top = current_room * np.abs(scalars(streamfunction)).max(axis=-1)  # [...]
    yield from current_frames(topology, points, areas, chain([first], iterator), centre, width,
                              field, fibres, shown, peak, stream_top)


def animate_induction(topology: Topology, points: Point, areas: Scalar,
                      centre: Point, width: float, field: Bivector,
                      fibres: tuple[Point, ...], shown: np.ndarray,
                      states: Iterable[tuple]) -> Iterator[np.ndarray]:
    """Yields inductive spin-up and braking frames on a fixed colour scale up to
    `induction_current_peak` so currents fade in as modal amplitudes grow from rest."""
    yield from current_frames(topology, points, areas, states, centre, width, field, fibres, shown,
                              induction_current_peak, np.zeros(len(shown)))


def current_frames(topology: Topology, points: Point, areas: Scalar,
                   states: Iterable[tuple], centre: Point, width: float,
                   field: Bivector, fibres: tuple[Point, ...], shown: np.ndarray,
                   peak: float, stream_top: np.ndarray) -> Iterator[np.ndarray]:
    """Yields RGB frames of rotating discs, body-fixed fibre guides, rim markers, and current contours."""
    extent = float(points.antinorm().batch().max().to_array())
    guides = concatenate(fibres)                                             # [..., n_samples] Point-
    family = np.concatenate([np.full(len(curves), index) for index, curves in enumerate(fibres)])
    shown_curves = shown[:, family]                                          # [..., ...]
    outline = topology.border
    n_panels = len(shown)

    figure = plt.figure(figsize=(3.3 * n_panels, 3.6), dpi=animation_dpi)
    panels = [disc_panel(figure, i, n_panels, extent) for i in range(n_panels)]
    views = [offset_panel(ax, mv.x, mv.y) for ax in panels]
    magnets = [magnet(centre, width, field, view) for view in views]
    textures = [overlay(ax, [np.zeros(view.shape + (4,))]) for ax, view in zip(panels, views)]
    for state in states:
        orientation, _, world_vertices, _, current, streamfunction = state[:6]
        strength = current.antinorm()                                        # [...] Scalar+[F]
        magnitude = scalars(strength)                                        # [..., F]
        stream = scalars(streamfunction)                                     # [..., V]
        peak = max(peak, magnitude.max())
        stream_top = np.maximum(stream_top, np.abs(stream).max(axis=-1))
        spacings = np.where(stream_top > 0, stream_top, 1.0) / n_current_levels  # [...]
        alpha = np.divide(magnitude.max(axis=-1), peak, out=np.zeros(len(magnitude)),
                          where=peak > 0) ** streamline_gamma
        current_scale = PowerNorm(current_gamma, vmin=0, vmax=peak)
        speed = scalars(vertex_average(topology, areas, strength))           # [..., V]
        places = world_vertices.batch()                                      # [..., V] Point-
        threads = orientation[:, None, None] >> guides                       # [..., ..., n_samples] Point-
        marks = rotation_marks(orientation, extent)                          # [...] Point-
        for i, (view, texture) in enumerate(zip(views, textures)):
            colours = current_colours(current_scale(speed[i]))[:, :3]
            ink = np.append(path_ink[:3], alpha[i])
            paths = contours.Lines(spacings[i], spacings[i] / 2, ink, 1.0)
            texture.set_data(drawing.composited([
                contours.picture(places[i], topology, stream[i], colours, view, lines=(paths,)),
                fibre_strokes(threads[i], shown_curves[i], view),
                simplices.strokes(places[i], outline, rim_colour, rim_width, view),
                *magnets[i], *rotation_marker(marks[i:i + 1], view)]))
        yield capture(figure)
    plt.close(figure)


def heat_frames(topology: Topology, points: Point, areas: Scalar,
                centre: Point, width: float, field: Bivector,
                fibres: tuple[Point, ...], shown: np.ndarray,
                peak: float, states: Iterable[tuple]) -> Iterator[np.ndarray]:
    """Yields accumulated Joule heat per unit area carried by the rotating triangles on a fixed
    colour scale up to `peak`."""
    heat_scale = PowerNorm(heat_gamma, vmin=0, vmax=peak)
    extent = float(points.antinorm().batch().max().to_array())
    guides = concatenate(fibres)                                             # [..., n_samples] Point-
    family = np.concatenate([np.full(len(curves), index) for index, curves in enumerate(fibres)])
    shown_curves = shown[:, family]                                          # [..., ...]
    outline = topology.border
    n_panels = len(shown)

    figure = plt.figure(figsize=(3.3 * n_panels, 3.6), dpi=animation_dpi)
    panels = [disc_panel(figure, i, n_panels, extent) for i in range(n_panels)]
    views = [offset_panel(ax, mv.x, mv.y) for ax in panels]
    magnets = [magnet(centre, width, field, view) for view in views]
    textures = [overlay(ax, [np.zeros(view.shape + (4,))]) for ax, view in zip(panels, views)]
    for state in states:
        orientation, _, world_vertices, heat = state[:4]
        densities = scalars(vertex_average(topology, areas, heat / areas))   # [..., V]
        places = world_vertices.batch()                                      # [..., V] Point-
        threads = orientation[:, None, None] >> guides                       # [..., ..., n_samples] Point-
        marks = rotation_marks(orientation, extent)                          # [...] Point-
        for i, (view, texture) in enumerate(zip(views, textures)):
            colours = heat_colours(heat_scale(densities[i]))[:, :3]
            texture.set_data(drawing.composited([
                contours.picture(places[i], topology, densities[i], colours, view),
                fibre_strokes(threads[i], shown_curves[i], view),
                simplices.strokes(places[i], outline, rim_colour, rim_width, view),
                *magnets[i], *rotation_marker(marks[i:i + 1], view)]))
        yield capture(figure)
    plt.close(figure)


def animate_heat(topology: Topology, points: Point, areas: Scalar,
                 centre: Point, width: float, field: Bivector,
                 fibres: tuple[Point, ...], shown: np.ndarray,
                 states: Iterable[tuple]) -> Iterator[np.ndarray]:
    """Yields accumulated Joule heat per unit area across quasistatic braking on a fixed scale up to
    `braking_heat_peak`."""
    yield from heat_frames(topology, points, areas, centre, width, field, fibres, shown,
                           braking_heat_peak, states)


def animate_induction_heat(topology: Topology, points: Point, areas: Scalar,
                           centre: Point, width: float, field: Bivector,
                           fibres: tuple[Point, ...], shown: np.ndarray,
                           states: Iterable[tuple]) -> Iterator[np.ndarray]:
    """Yields accumulated Joule heat per unit area across inductive spin-up and braking on a fixed
    scale up to `induction_heat_peak`."""
    yield from heat_frames(topology, points, areas, centre, width, field, fibres, shown,
                           induction_heat_peak, states)
