"""Planar rendering for flexible girders, swinging chains, and buckling columns."""

from collections.abc import Iterable, Iterator
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from numga import Extensor, concatenate, stack
from numga.algebras import PGA2D as ga
from numga.render import Camera, drawing, simplices
from numga.topology import Topology
from examples.util import charts
from examples.util.animation import capture
from examples.util.panels import overlay, panel

Point = ga.gatype.antivector()                                             # Antivector-


colours = ("#2563eb", "#0d9488", "#d97706", "#9333ea")
cross_colour, label_colour, critical_colour = "0.85", "0.4", "#dc2626"
rest_colour, pivot_colour, fixed_colour = "#cbd5e1", "#334155", "#94a3b8"
# Stroke widths in pixels for rest girders, displaced modes, chain links, and beam girders, plus pivot marker radius.
rest_width, mode_width, chain_width, beam_width, pivot_radius = 1.1, 1.7, 1.8, 0.7, 3.5
mode_margin = 0.05
chain_room, chain_above = 1.12, 0.15
beam_room, beam_end_room = 1.2, 0.005


def point_coordinates(points: Point) -> Extensor:
    """Pairing unit-weight `points` against the coordinate planes `x` and `y` reads their planar
    offsets (`[2, ...] Scalar+`)."""
    mv = points.context.multivector
    return stack((mv.x, mv.y))[:, None] & points.batch().reshape(-1)       # [2, ...] Scalar+


def rgb_colours(names) -> np.ndarray:
    """Named colour strings convert to floating-point RGB triples (`[..., 3]`)."""
    return np.array([to_rgb(colour) for colour in np.ravel(names)]).reshape(np.shape(names) + (3,))


def bodies_topology(n_bodies: int, n_vertices: int, edges: np.ndarray) -> Topology:
    """Shifting `edges` by `n_vertices` per girder combines `n_bodies` identical girders into one
    bar complex."""
    offsets = n_vertices * np.arange(n_bodies)                             # [n_bodies]
    return Topology(n_bodies * n_vertices, (edges[None] + offsets[:, None, None]).reshape(-1, 2))


def mode_shapes(rest: Point, deformed: Point, edges: np.ndarray) -> plt.Figure:
    """Overlays each displaced vibration mode in `deformed` onto the grey `rest` girder."""
    figure = plt.figure(figsize=(8, 5), layout="constrained")
    n_columns = 2
    shapes = deformed.batch()                                              # [..., V] Antivector-
    n_rows = (len(shapes) + n_columns - 1) // n_columns
    panels = []
    bounds = []
    for i, shape in enumerate(shapes):
        axes = figure.add_subplot(n_rows, n_columns, i + 1)
        bounds_points = concatenate([point_coordinates(rest), point_coordinates(shape)], axis=-1)  # [2, ...] Scalar+
        low, high = bounds_points.min(axis=-1), bounds_points.max(axis=-1)
        padding = mode_margin * (high - low)
        xlim = stack((low[0] - padding[0], high[0] + padding[0]))
        ylim = stack((low[1] - padding[1], high[1] + padding[1]))
        axes.set(xlim=xlim.to_array(), ylim=ylim.to_array(), aspect="equal")
        axes.set_axis_off()
        panels.append(axes)
        bounds.append((xlim, ylim))
    girder = Topology(rest.batch().shape[0], edges)
    mv = rest.context.multivector
    for i, (axes, (xlim, ylim), shape) in enumerate(zip(panels, bounds, shapes)):
        camera = panel(axes, mv.x, mv.y, mv.w, xlim=xlim, ylim=ylim)
        overlay(
            axes,
            [
                simplices.strokes(rest.batch(), girder, rgb_colours(rest_colour), rest_width, camera),
                simplices.strokes(shape, girder, rgb_colours(colours[i % len(colours)]), mode_width, camera),
            ],
        )
    return figure


def chain_scene(edges: np.ndarray, reference: Point, lower: Point, upper: Point) -> tuple[plt.Figure, plt.Axes, Camera]:
    """Framing `axes` between corner points `lower` and `upper` builds the planar `Camera` for the
    chain."""
    figure, axes = plt.subplots(figsize=(6, 4), layout="constrained")
    mv = lower.context.multivector
    corners = stack((lower, upper))                                        # [2] Antivector-
    xlim, ylim = mv.x & corners, mv.y & corners
    axes.set_aspect("equal")
    axes.set_axis_off()
    return figure, axes, panel(axes, mv.x, mv.y, mv.w, xlim=xlim, ylim=ylim)


def chain_composite(points: Point, edges: np.ndarray, reference: Point, camera: Camera) -> np.ndarray:
    """Composites the moving links of the first case in distinct colours with a marker at the root
    pivot."""
    bodies = points[0, 1:].batch()                                         # [..., V] Antivector-
    n_bodies, n_vertices = bodies.shape
    pivot = reference[0, 1:].batch()[0, 1]                                 # [] Antivector-
    body_colours = rgb_colours(np.asarray(colours)[np.arange(n_bodies) % len(colours)])
    return drawing.composited([
        simplices.strokes(
            bodies.reshape(-1),
            bodies_topology(n_bodies, n_vertices, edges),
            np.repeat(body_colours, len(edges), axis=0),
            chain_width,
            camera,
        ),
        simplices.markers(pivot[None], rgb_colours(pivot_colour), pivot_radius, camera),
    ])


def swinging_chain(edges: np.ndarray, reference: Point, states: Iterable) -> Iterator[np.ndarray]:
    """Yields RGB frames of the swinging chain inside a fixed viewport sized from the initial reach
    around the root pivot."""
    frames = (state[0] if isinstance(state, tuple) else state for state in states)
    first = next(frames)
    mv = first.context.multivector
    pivot = reference[0, 1:].batch()[0, 1]                                 # [] Antivector-
    distances = (first[0, 1:].batch() - pivot).dual().norm().reshape(-1)   # [...] Scalar+
    extent = chain_room * distances.max()                                  # [] Scalar+
    lower = pivot - (mv.x.dual() + mv.y.dual()) * extent                   # [] Antivector-
    upper = pivot + (mv.x.dual() + mv.y.dual() * chain_above) * extent     # [] Antivector-
    figure, axes, camera = chain_scene(edges, reference, lower, upper)
    image = overlay(axes, [chain_composite(first, edges, reference, camera)])
    for points in chain([first], frames):
        image.set_data(chain_composite(points, edges, reference, camera))
        yield capture(figure)
    plt.close(figure)


def chain_figure(points: Point, edges: np.ndarray) -> plt.Figure:
    """Renders a single still of the hanging chain framed around its current bounding box."""
    mv = points.context.multivector
    frame = stack((mv.x, mv.y))                                            # [2] Vector+
    positions = points[0, 1:].batch().reshape(-1)                          # [...] Antivector-
    coords = frame[:, None] & positions                                    # [2, ...] Scalar+
    low = frame & positions[coords.argmin(axis=-1)]                        # [2] Scalar+
    high = frame & positions[coords.argmax(axis=-1)]                       # [2] Scalar+
    spans = high - low                                                     # [2] Scalar+
    margin = spans.max() * 0.06                                            # [] Scalar+
    lower = mv.w.dual() + (frame.dual() * (low - margin)).sum()            # [] Antivector-
    upper = mv.w.dual() + (frame.dual() * (high + margin)).sum()           # [] Antivector-
    figure, axes, camera = chain_scene(edges, points, lower, upper)
    overlay(axes, [chain_composite(points, edges, points, camera)])
    return figure


def beam_animation(edges: np.ndarray, states: Iterable) -> Iterator[np.ndarray]:
    """Yields RGB frames of the compressed column with grey fixed supports and coloured interior
    girders."""
    frames = (state[0] if isinstance(state, tuple) else state for state in states)
    first = next(frames)
    mv = first.context.multivector
    positions = first[0].batch()                                           # [..., V] Antivector-
    n_bodies, n_vertices = positions.shape
    body_colours = rgb_colours(np.where(
        (np.arange(n_bodies) == 0) | (np.arange(n_bodies) == n_bodies - 1),
        fixed_colour,
        np.asarray(colours)[np.arange(n_bodies) % len(colours)],
    ))
    girders = bodies_topology(n_bodies, n_vertices, edges)
    horizontal = (mv.x & positions).reshape(-1)                            # [...] Scalar+
    heights = (mv.y & positions).abs().reshape(-1)                         # [...] Scalar+
    left, right = horizontal.min(), horizontal.max()
    extent = beam_room * max(heights.max(), (right - left) / 40)
    beyond = beam_end_room * (right - left)
    figure, axes = plt.subplots(figsize=(10, 1.6), dpi=60, layout="constrained")
    xlim, ylim = stack((left - beyond, right + beyond)), stack((-extent, extent))
    axes.set_aspect("equal")
    axes.set_axis_off()
    camera = panel(axes, mv.x, mv.y, mv.w, xlim=xlim, ylim=ylim)
    image = overlay(axes, [np.zeros(camera.shape + (4,))])
    for points in chain([first], frames):
        image.set_data(drawing.composited([
            simplices.strokes(
                points[0].batch().reshape(-1),
                girders,
                np.repeat(body_colours, len(edges), axis=0),
                beam_width,
                camera,
            ),
        ]))
        yield capture(figure)
    plt.close(figure)


@charts.style()
def buckling_figure(displacements: Extensor | np.ndarray, midspans: Point, critical: Extensor | float) -> plt.Figure:
    """Plots midspan lateral deflection against prescribed end displacement alongside the Euler
    critical threshold."""
    deflection = (midspans.context.multivector.y & midspans[:, 0].batch()).abs().to_array()
    x_values = displacements.to_array() if isinstance(displacements, Extensor) else displacements
    threshold = float(critical.to_array()) if isinstance(critical, Extensor) else critical
    figure, axes = charts.subplots(400, 220, layout="constrained")
    axes.plot(x_values, deflection, color="#0f172a")
    axes.axhline(0, color=cross_colour, linewidth=charts.FRAME_WIDTH, zorder=0)
    axes.axvline(0, color=cross_colour, linewidth=charts.FRAME_WIDTH, zorder=0)
    axes.text(1, 0, "end displacement", transform=axes.get_yaxis_transform(), ha="right", va="top", color=label_colour)
    axes.text(
        0, 1, "midspan deflection ", transform=axes.get_xaxis_transform(),
        ha="right", va="bottom", color=label_colour, rotation=90, rotation_mode="anchor",
    )
    axes.axvline(threshold, linestyle=":", color=critical_colour)
    axes.text(threshold, 1, " critical", transform=axes.get_xaxis_transform(), ha="left", va="top", color=critical_colour)
    axes.set_axis_off()
    return figure
