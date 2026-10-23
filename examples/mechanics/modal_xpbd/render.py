"""Flexible girders and their tip motion."""

from collections.abc import Iterable, Iterator
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from numga.algebras import PGA2D as ga
from examples.util.animation import capture

Point = ga.gatype.antivector()


COLOURS = ("#2563eb", "#0d9488", "#d97706", "#9333ea")
CROSS_COLOUR, LABEL_COLOUR, CRITICAL_COLOUR = "0.85", "0.4", "#dc2626"
# How many times the first frame's reach from the pivot the swinging chain's view spans to either side and below, room
# for the swing, and the share of that it spans above; how many times the first frame's farthest point from its axis,
# or a fortieth of its length, the beam's view spans above and below the axis, room for the buckling, and the share of
# its length the view runs on past either end, room for the bars' width.
CHAIN_ROOM, CHAIN_ABOVE = 1.12, 0.15
BEAM_ROOM, BEAM_END_ROOM = 1.2, 0.005


def mode_shapes(rest: Point, deformed: Point, edges: np.ndarray) -> plt.Figure:
    reference = rest.dual().cast(ga.subspace("x y")).kernel
    positions = deformed.dual().cast(ga.subspace("x y")).kernel
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


def history(points: Point, damping: np.ndarray, frame_dt: float) -> plt.Figure:
    vertices = points.batch()
    tips = (vertices[..., -1, -2] + vertices[..., -1, -1]) / 2
    positions = tips.dual().cast(ga.subspace("x y")).kernel
    time = np.arange(len(positions)) * frame_dt
    figure, axes = plt.subplots(figsize=(6.5, 3), layout="constrained")
    for i, ratio in enumerate(damping):
        axes.plot(time, positions[:, i, 1], color=COLOURS[i % len(COLOURS)], label=rf"$\zeta={ratio:g}$")
    axes.axhline(0, color="#cbd5e1", linewidth=0.8)
    axes.set(xlabel="Time", ylabel="Tip displacement")
    axes.spines[["top", "right"]].set_visible(False)
    axes.legend(frameon=False, title="Local modal damping")
    return figure


def chain_scene(edges: np.ndarray, reference: Point, lower: np.ndarray,
                upper: np.ndarray) -> tuple[plt.Figure, LineCollection]:
    """The chain's figure between the corners `lower` and `upper`, its pivot marked, its bars still to be set."""
    rest = reference[0, 1:].dual().cast(ga.subspace("x y")).kernel
    pivot = rest[0, 1]
    body_colours = np.asarray(COLOURS)[np.arange(len(rest)) % len(COLOURS)]
    edge_colours = np.repeat(body_colours, len(edges))
    figure, axes = plt.subplots(figsize=(6, 4), layout="constrained")
    bars = LineCollection([], colors=edge_colours, linewidths=1.3)
    axes.add_collection(bars)
    axes.plot(*pivot, marker="o", color="#334155", markersize=5)
    axes.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]))
    axes.set_aspect("equal")
    axes.set_axis_off()
    return figure, bars


def update_chain(points: Point, edges: np.ndarray, bars: LineCollection) -> None:
    positions = points[0, 1:].dual().cast(ga.subspace("x y")).kernel
    bars.set_segments(positions[:, edges].reshape(-1, 2, 2))


def swinging_chain(frames: Iterable[Point], edges: np.ndarray, reference: Point,
                   room: float = CHAIN_ROOM) -> Iterator[np.ndarray]:
    """Frames of the swinging chain, the first case, in a view spanning the first frame's reach from the pivot `room`
    times over, to either side and below, and a share of that above."""
    frames = iter(frames)
    first = next(frames)
    pivot = reference[0, 1:].dual().cast(ga.subspace("x y")).kernel[0, 1]
    positions = first[0, 1:].dual().cast(ga.subspace("x y")).kernel     # [bodies, vertices, 2]
    reach = room * np.linalg.norm(positions - pivot, axis=-1).max()
    figure, bars = chain_scene(edges, reference, pivot - reach, pivot + reach * np.array([1, CHAIN_ABOVE]))
    for points in chain([first], frames):
        update_chain(points, edges, bars)
        yield capture(figure)
    plt.close(figure)


def draw_chain(points: Point, edges: np.ndarray) -> plt.Figure:
    """The chain at one moment."""
    positions = points[0, 1:].dual().cast(ga.subspace("x y")).kernel
    lower, upper = positions.min(axis=(0, 1)), positions.max(axis=(0, 1))
    margin = (upper - lower).max() * 0.06
    figure, bars = chain_scene(edges, points, lower - margin, upper + margin)
    update_chain(points, edges, bars)
    return figure


def beam(frames: Iterable[Point], edges: np.ndarray, room: float = BEAM_ROOM,
         end_room: float = BEAM_END_ROOM) -> Iterator[np.ndarray]:
    """Frames of the beam, the first case: moving girders coloured, fixed ones grey, in a view along the first frame,
    running on past either end by `end_room` of its length, that spans above and below its axis `room` times its
    farthest point from the axis or a fortieth of its length, whichever is more."""
    frames = iter(frames)
    first = next(frames)
    positions = first[0].dual().cast(ga.subspace("x y")).kernel         # [bodies, vertices, 2]
    bodies = positions.shape[0]
    colours = np.where((np.arange(bodies) == 0) | (np.arange(bodies) == bodies - 1), "#94a3b8",
                       np.asarray(COLOURS)[np.arange(bodies) % len(COLOURS)])
    lower, upper = positions.min(axis=(0, 1)), positions.max(axis=(0, 1))
    reach = room * max(np.abs(positions[..., 1]).max(), (upper[0] - lower[0]) / 40)
    beyond = end_room * (upper[0] - lower[0])
    figure, axes = plt.subplots(figsize=(10, 1.6), dpi=60, layout="constrained")
    bars = LineCollection([], colors=np.repeat(colours, len(edges)), linewidths=0.8)
    axes.add_collection(bars)
    axes.set(xlim=(lower[0] - beyond, upper[0] + beyond), ylim=(-reach, reach))
    axes.set_aspect("equal")
    axes.set_axis_off()
    for points in chain([first], frames):
        frame = points[0].dual().cast(ga.subspace("x y")).kernel
        bars.set_segments(frame[:, edges].reshape(-1, 2, 2))
        yield capture(figure)
    plt.close(figure)


def buckling(displacements: np.ndarray, midspans: Point, critical: float) -> plt.Figure:
    """Midspan deflection against end displacement, the first case, with the critical value."""
    deflection = np.abs(midspans[:, 0].dual().cast(ga.subspace("x y")).kernel[:, 1])
    figure, axes = plt.subplots(figsize=(6.5, 3), layout="constrained")
    axes.plot(displacements, deflection, color="#0f172a")
    # A faint cross through the origin, without scales, its lines named at their ends, each along itself; the
    # critical value named at the top of its line.
    axes.axhline(0, color=CROSS_COLOUR, linewidth=0.7, zorder=0)
    axes.axvline(0, color=CROSS_COLOUR, linewidth=0.7, zorder=0)
    axes.text(1, 0, "end displacement", transform=axes.get_yaxis_transform(), ha="right", va="top", fontsize=9,
              color=LABEL_COLOUR)
    axes.text(0, 1, "midspan deflection ", transform=axes.get_xaxis_transform(), ha="right", va="bottom", fontsize=9,
              color=LABEL_COLOUR, rotation=90, rotation_mode="anchor")
    axes.axvline(critical, linestyle=":", color=CRITICAL_COLOUR)
    axes.text(critical, 1, " critical", transform=axes.get_xaxis_transform(), ha="left", va="top", fontsize=9,
              color=CRITICAL_COLOUR)
    axes.set_axis_off()
    return figure
