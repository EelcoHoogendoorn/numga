"""Flexible girders and their tip motion."""

from collections.abc import Iterable, Iterator
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from numga import Extensor, concatenate, stack
from numga.algebras import PGA2D as ga
from examples.util import charts
from examples.util.animation import capture
from examples.util.wgsl import simplices
from examples.util.wgsl.camera import Camera, flat_panel

Point = ga.gatype.antivector()  # Antivector-


COLOURS = ("#2563eb", "#0d9488", "#d97706", "#9333ea")
CROSS_COLOUR, LABEL_COLOUR, CRITICAL_COLOUR = "0.85", "0.4", "#dc2626"
REST_COLOUR, PIVOT_COLOUR, FIXED_COLOUR = "#cbd5e1", "#334155", "#94a3b8"
# In pixels: the widths of a mode's rest shape and of its moved shape, of the chain's bars and of the beam's, and the
# radius of the pivot.
REST_WIDTH, MODE_WIDTH, CHAIN_WIDTH, BEAM_WIDTH, PIVOT_RADIUS = 1.1, 1.7, 1.8, 0.7, 3.5
# The margin around the mode shapes, as a share of their extent either way.
MODE_MARGIN = 0.05
# How many times the first frame's reach from the pivot the swinging chain's view spans to either side and below, room
# for the swing, and the share of that it spans above; how many times the first frame's farthest point from its axis,
# or a fortieth of its length, the beam's view spans above and below the axis, room for the buckling, and the share of
# its length the view runs on past either end, room for the bars' width.
CHAIN_ROOM, CHAIN_ABOVE = 1.12, 0.15
BEAM_ROOM, BEAM_END_ROOM = 1.2, 0.005


def readings(points: Point) -> Extensor:
    """Points of unit weight read across and up, `[2, points]`: their pairings with the planes x = 0 and y = 0."""
    mv = points.context.multivector
    return stack((mv.x, mv.y))[:, None] & points.batch().reshape(-1)


def rgb(colours) -> np.ndarray:
    """Colours by name as red, green and blue `[..., 3]`."""
    return np.array([to_rgb(colour) for colour in np.ravel(colours)]).reshape(np.shape(colours) + (3,))


def bodies_mesh(n_bodies: int, n_vertices: int, edges: np.ndarray) -> simplices.Mesh:
    """The bars of bodies alike, their vertices numbered body by body, as one mesh of edges."""
    offsets = n_vertices * np.arange(n_bodies)
    return simplices.Mesh(n_bodies * n_vertices, (edges[None] + offsets[:, None, None]).reshape(-1, 2))


def mode_shapes(rest: Point, deformed: Point, edges: np.ndarray) -> plt.Figure:
    figure = plt.figure(figsize=(8, 5), layout="constrained")
    columns = 2
    shapes = deformed.batch()                                                # [modes, vertices] Point-
    rows = (len(shapes) + columns - 1) // columns
    panels = []
    for i, shape in enumerate(shapes):
        axes = figure.add_subplot(rows, columns, i + 1)
        drawn = concatenate([readings(rest), readings(shape)], axis=-1)      # [2, points] Scalar+
        low, high = drawn.min(axis=-1).to_array(), drawn.max(axis=-1).to_array()
        padding = MODE_MARGIN * (high - low)
        axes.set(xlim=(low[0] - padding[0], high[0] + padding[0]), ylim=(low[1] - padding[1], high[1] + padding[1]),
                 aspect="equal")
        axes.set_axis_off()
        panels.append(axes)
    girder = simplices.Mesh(rest.batch().shape[0], edges)
    for i, (axes, shape) in enumerate(zip(panels, shapes)):
        camera = flat_panel(axes, ga)
        simplices.overlay(axes, [simplices.strokes(rest.batch(), girder, rgb(REST_COLOUR), REST_WIDTH, camera),
                                 simplices.strokes(shape, girder, rgb(COLOURS[i % len(COLOURS)]), MODE_WIDTH,
                                                   camera)])
    return figure


@charts.style()
def history(points: Point, damping: np.ndarray, frame_dt: float) -> plt.Figure:
    vertices = points.batch()
    tips = (vertices[..., -1, -2] + vertices[..., -1, -1]) / 2
    # The tip's displacement, its reading up, by the plane y = 0.
    heights = (vertices.context.multivector.y & tips).to_array()            # [steps, cases]
    time = np.arange(len(heights)) * frame_dt
    # Three fifths of the column: a few decaying curves, and their legend beside them.
    figure, axes = charts.subplots(480, 260, layout="constrained")
    for i, ratio in enumerate(damping):
        axes.plot(time, heights[:, i], color=COLOURS[i % len(COLOURS)], label=rf"$\zeta={ratio:g}$")
    axes.axhline(0, color="#cbd5e1", linewidth=charts.FRAME_WIDTH)
    axes.set(xlabel="Time", ylabel="Tip displacement")
    axes.spines[["top", "right"]].set_visible(False)
    figure.legend(*axes.get_legend_handles_labels(), frameon=False, title="Local modal damping",
                  loc="outside right upper")
    return figure


def chain_scene(edges: np.ndarray, reference: Point, lower: Point,
                upper: Point) -> tuple[plt.Figure, plt.Axes, Camera]:
    """The chain's figure between the corners `lower` and `upper`, and its camera."""
    figure, axes = plt.subplots(figsize=(6, 4), layout="constrained")
    mv = lower.context.multivector
    corners = stack((lower, upper))                                          # [2] Point-
    axes.set(xlim=(mv.x & corners).to_array(), ylim=(mv.y & corners).to_array())
    axes.set_aspect("equal")
    axes.set_axis_off()
    return figure, axes, flat_panel(axes, ga)


def chain_drawn(points: Point, edges: np.ndarray, reference: Point, camera: Camera) -> np.ndarray:
    """The chain's bars at one moment, the first case's bodies after the first, each in its colour, with its pivot
    marked."""
    bodies = points[0, 1:].batch()                                           # [bodies, vertices] Point-
    n_bodies, n_vertices = bodies.shape
    pivot = reference[0, 1:].batch()[0, 1]                                   # [] Point-
    body_colours = rgb(np.asarray(COLOURS)[np.arange(n_bodies) % len(COLOURS)])
    return simplices.composited([
        simplices.strokes(bodies.reshape(-1), bodies_mesh(n_bodies, n_vertices, edges),
                          np.repeat(body_colours, len(edges), axis=0), CHAIN_WIDTH, camera),
        simplices.markers(pivot[None], rgb(PIVOT_COLOUR), PIVOT_RADIUS, camera)])


def swinging_chain(frames: Iterable[Point], edges: np.ndarray, reference: Point) -> Iterator[np.ndarray]:
    """Frames of the swinging chain, the first case, in a view spanning the first frame's reach from the pivot `CHAIN_ROOM`
    times over, to either side and below, and a share of that above."""
    frames = iter(frames)
    first = next(frames)
    mv = first.context.multivector
    pivot = reference[0, 1:].batch()[0, 1]
    distances = (first[0, 1:].batch() - pivot).dual().norm().reshape(-1)
    extent = CHAIN_ROOM * distances.max()
    lower = pivot - (mv.x.dual() + mv.y.dual()) * extent
    upper = pivot + (mv.x.dual() + mv.y.dual() * CHAIN_ABOVE) * extent
    figure, axes, camera = chain_scene(edges, reference, lower, upper)
    image = simplices.overlay(axes, [chain_drawn(first, edges, reference, camera)])
    for points in chain([first], frames):
        image.set_data(chain_drawn(points, edges, reference, camera))
        yield capture(figure)
    plt.close(figure)


def draw_chain(points: Point, edges: np.ndarray) -> plt.Figure:
    """The chain at one moment."""
    mv = points.context.multivector
    frame = stack((mv.x, mv.y))
    positions = points[0, 1:].batch().reshape(-1)
    readings = frame[:, None] & positions
    low = frame & positions[readings.argmin(axis=-1)]
    high = frame & positions[readings.argmax(axis=-1)]
    spans = high - low
    margin = spans.max() * 0.06
    lower = mv.w.dual() + (frame.dual() * (low - margin)).sum()
    upper = mv.w.dual() + (frame.dual() * (high + margin)).sum()
    figure, axes, camera = chain_scene(edges, points, lower, upper)
    simplices.overlay(axes, [chain_drawn(points, edges, points, camera)])
    return figure


def beam(frames: Iterable[Point], edges: np.ndarray) -> Iterator[np.ndarray]:
    """Frames of the beam, the first case: moving girders coloured, fixed ones grey, in a view along the first frame,
    running on past either end by `BEAM_END_ROOM` of its length, that spans above and below its axis `BEAM_ROOM` times its
    farthest point from the axis or a fortieth of its length, whichever is more."""
    frames = iter(frames)
    first = next(frames)
    mv = first.context.multivector
    positions = first[0].batch()                                             # [bodies, vertices] Point-
    bodies = positions.shape[0]
    colours = rgb(np.where((np.arange(bodies) == 0) | (np.arange(bodies) == bodies - 1), FIXED_COLOUR,
                           np.asarray(COLOURS)[np.arange(bodies) % len(COLOURS)]))
    girders = bodies_mesh(bodies, positions.shape[1], edges)
    along = (mv.x & positions).reshape(-1)
    heights = (mv.y & positions).abs().reshape(-1)
    left, right = along.min(), along.max()
    extent = BEAM_ROOM * max(heights.max(), (right - left) / 40)
    beyond = BEAM_END_ROOM * (right - left)
    figure, axes = plt.subplots(figsize=(10, 1.6), dpi=60, layout="constrained")
    axes.set(xlim=stack((left - beyond, right + beyond)).to_array(), ylim=stack((-extent, extent)).to_array())
    axes.set_aspect("equal")
    axes.set_axis_off()
    camera = flat_panel(axes, ga)
    image = simplices.overlay(axes, [np.zeros(camera.shape + (4,))])
    for points in chain([first], frames):
        image.set_data(simplices.strokes(points[0].batch().reshape(-1), girders, np.repeat(colours, len(edges), axis=0),
                                         BEAM_WIDTH, camera))
        yield capture(figure)
    plt.close(figure)


@charts.style()
def buckling(displacements: np.ndarray, midspans: Point, critical: float) -> plt.Figure:
    """Midspan deflection against end displacement, the first case, with the critical value."""
    deflection = (midspans.context.multivector.y & midspans[:, 0].batch()).abs().to_array()
    # Half the column: one curve, its cross and the critical value named.
    figure, axes = charts.subplots(400, 220, layout="constrained")
    axes.plot(displacements, deflection, color="#0f172a")
    # A faint cross through the origin, without scales, its lines named at their ends, each along itself; the
    # critical value named at the top of its line.
    axes.axhline(0, color=CROSS_COLOUR, linewidth=charts.FRAME_WIDTH, zorder=0)
    axes.axvline(0, color=CROSS_COLOUR, linewidth=charts.FRAME_WIDTH, zorder=0)
    axes.text(1, 0, "end displacement", transform=axes.get_yaxis_transform(), ha="right", va="top", color=LABEL_COLOUR)
    axes.text(0, 1, "midspan deflection ", transform=axes.get_xaxis_transform(), ha="right", va="bottom",
              color=LABEL_COLOUR, rotation=90, rotation_mode="anchor")
    axes.axvline(critical, linestyle=":", color=CRITICAL_COLOUR)
    axes.text(critical, 1, " critical", transform=axes.get_xaxis_transform(), ha="left", va="top",
              color=CRITICAL_COLOUR)
    axes.set_axis_off()
    return figure
