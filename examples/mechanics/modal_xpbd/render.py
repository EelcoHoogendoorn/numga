"""Flexible girders and their tip motion."""

from collections.abc import Iterable, Iterator
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from numga import stack
from numga.algebras import PGA2D as ga
from examples.util import charts
from examples.util.animation import capture

Point = ga.gatype.antivector()  # Antivector-


COLOURS = ("#2563eb", "#0d9488", "#d97706", "#9333ea")
CROSS_COLOUR, LABEL_COLOUR, CRITICAL_COLOUR = "0.85", "0.4", "#dc2626"
# How many times the first frame's reach from the pivot the swinging chain's view spans to either side and below, room
# for the swing, and the share of that it spans above; how many times the first frame's farthest point from its axis,
# or a fortieth of its length, the beam's view spans above and below the axis, room for the buckling, and the share of
# its length the view runs on past either end, room for the bars' width.
CHAIN_ROOM, CHAIN_ABOVE = 1.12, 0.15
BEAM_ROOM, BEAM_END_ROOM = 1.2, 0.005


def coordinates(points: Point) -> np.ndarray:
    """Coordinates of points of unit weight, `[..., 2]`: their pairings with the planes x = 0 and y = 0."""
    mv = points.context.multivector
    return (stack((mv.x, mv.y)) & points.batch()[..., None]).to_array()


def mode_shapes(rest: Point, deformed: Point, edges: np.ndarray) -> plt.Figure:
    reference = coordinates(rest)
    positions = coordinates(deformed)
    figure = plt.figure(figsize=(8, 5), layout="constrained")
    columns = 2
    rows = (len(positions) + columns - 1) // columns
    for i, points in enumerate(positions):
        axes = figure.add_subplot(rows, columns, i + 1)
        axes.add_collection(LineCollection(reference[edges], colors="#cbd5e1", linewidths=0.8))
        axes.add_collection(LineCollection(points[edges], colors=COLOURS[i % len(COLOURS)], linewidths=1.2))
        axes.autoscale()
        axes.set_aspect("equal")
        axes.set_axis_off()
    return figure


@charts.style()
def history(points: Point, damping: np.ndarray, frame_dt: float) -> plt.Figure:
    vertices = points.batch()
    tips = (vertices[..., -1, -2] + vertices[..., -1, -1]) / 2
    positions = coordinates(tips)
    time = np.arange(len(positions)) * frame_dt
    # Three fifths of the column: a few decaying curves, and their legend beside them.
    figure, axes = charts.subplots(480, 260, layout="constrained")
    for i, ratio in enumerate(damping):
        axes.plot(time, positions[:, i, 1], color=COLOURS[i % len(COLOURS)], label=rf"$\zeta={ratio:g}$")
    axes.axhline(0, color="#cbd5e1", linewidth=charts.FRAME_WIDTH)
    axes.set(xlabel="Time", ylabel="Tip displacement")
    axes.spines[["top", "right"]].set_visible(False)
    figure.legend(*axes.get_legend_handles_labels(), frameon=False, title="Local modal damping",
                  loc="outside right upper")
    return figure


def chain_scene(edges: np.ndarray, reference: Point, lower: Point,
                upper: Point) -> tuple[plt.Figure, LineCollection]:
    """The chain's figure between the corners `lower` and `upper`, its pivot marked, its bars still to be set."""
    rest = coordinates(reference[0, 1:])
    pivot = rest[0, 1]
    body_colours = np.asarray(COLOURS)[np.arange(len(rest)) % len(COLOURS)]
    edge_colours = np.repeat(body_colours, len(edges))
    figure, axes = plt.subplots(figsize=(6, 4), layout="constrained")
    bars = LineCollection([], colors=edge_colours, linewidths=1.3)
    axes.add_collection(bars)
    axes.plot(*pivot, marker="o", color="#334155", markersize=5)
    bounds = coordinates(stack((lower, upper)))
    axes.set(xlim=bounds[:, 0], ylim=bounds[:, 1])
    axes.set_aspect("equal")
    axes.set_axis_off()
    return figure, bars


def update_chain(points: Point, edges: np.ndarray, bars: LineCollection) -> None:
    positions = coordinates(points[0, 1:])
    bars.set_segments(positions[:, edges].reshape(-1, 2, 2))


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
    figure, bars = chain_scene(edges, reference, lower, upper)
    for points in chain([first], frames):
        update_chain(points, edges, bars)
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
    figure, bars = chain_scene(edges, points, lower, upper)
    update_chain(points, edges, bars)
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
    colours = np.where((np.arange(bodies) == 0) | (np.arange(bodies) == bodies - 1), "#94a3b8",
                       np.asarray(COLOURS)[np.arange(bodies) % len(COLOURS)])
    along = (mv.x & positions).reshape(-1)
    heights = (mv.y & positions).abs().reshape(-1)
    left, right = along.min(), along.max()
    extent = BEAM_ROOM * max(heights.max(), (right - left) / 40)
    beyond = BEAM_END_ROOM * (right - left)
    figure, axes = plt.subplots(figsize=(10, 1.6), dpi=60, layout="constrained")
    bars = LineCollection([], colors=np.repeat(colours, len(edges)), linewidths=0.8)
    axes.add_collection(bars)
    axes.set(xlim=stack((left - beyond, right + beyond)).to_array(), ylim=stack((-extent, extent)).to_array())
    axes.set_aspect("equal")
    axes.set_axis_off()
    for points in chain([first], frames):
        frame = coordinates(points[0])
        bars.set_segments(frame[:, edges].reshape(-1, 2, 2))
        yield capture(figure)
    plt.close(figure)


@charts.style()
def buckling(displacements: np.ndarray, midspans: Point, critical: float) -> plt.Figure:
    """Midspan deflection against end displacement, the first case, with the critical value."""
    deflection = np.abs(coordinates(midspans[:, 0])[:, 1])
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
