"""Flexible girders and their tip motion."""

from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from numga import stack
from examples.animation import capture, save_animation
from .core import Point, ga


COLOURS = ("#2563eb", "#0d9488", "#d97706", "#9333ea")


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


def draw(points: Point, edges: np.ndarray, damping: np.ndarray, reference: Point) -> plt.Figure:
    positions = points.dual().cast(ga.subspace("x y")).kernel
    rest = reference.dual().cast(ga.subspace("x y")).kernel[..., 1:, :, :]
    left, right = rest[..., 0].min(), rest[..., 0].max()
    span = right - left
    body_colours = np.asarray(COLOURS)[np.arange(positions.shape[1] - 1) % len(COLOURS)]
    edge_colours = np.repeat(body_colours, len(edges))
    figure = plt.figure(figsize=(8, 5), layout="constrained")
    for i, ratio in enumerate(damping):
        axes = figure.add_subplot(len(damping), 1, i + 1)
        segments = positions[i, 1:][:, edges].reshape(-1, 2, 2)
        axes.add_collection(LineCollection(segments, colors=edge_colours, linewidths=1.2))
        axes.axvline(left, color="#64748b", linewidth=2)
        axes.axhline(0, color="#cbd5e1", linewidth=0.5)
        axes.set(xlim=(left - span * 0.04, right + span * 0.04), ylim=(-span * 0.1, span * 0.1), ylabel=rf"$\zeta={ratio:g}$")
        axes.set_aspect("equal", adjustable="box")
        axes.spines[["top", "right", "left"]].set_visible(False)
        axes.set_yticks([])
    axes.set_xlabel("Position")
    return figure


def history(points: Point, damping: np.ndarray, frame_dt: float) -> plt.Figure:
    tips = (points[..., -1, -2] + points[..., -1, -1]) / 2
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


def animate(frames: Iterable[Point], edges: np.ndarray, damping: np.ndarray, reference: Point) -> list[np.ndarray]:
    images = []
    for points in frames:
        figure = draw(points, edges, damping, reference)
        images.append(capture(figure))
        plt.close(figure)
    return images


def chain_scene(edges: np.ndarray, reference: Point, trajectory: Point) -> tuple[plt.Figure, LineCollection]:
    rest = reference[0, 1:].dual().cast(ga.subspace("x y")).kernel
    positions = trajectory[:, 0, 1:].dual().cast(ga.subspace("x y")).kernel
    lower, upper = positions.min(axis=(0, 1, 2)), positions.max(axis=(0, 1, 2))
    margin = (upper - lower).max() * 0.06
    pivot = rest[0, 1]
    body_colours = np.asarray(COLOURS)[np.arange(len(rest)) % len(COLOURS)]
    edge_colours = np.repeat(body_colours, len(edges))
    figure, axes = plt.subplots(figsize=(6, 4), layout="constrained")
    bars = LineCollection([], colors=edge_colours, linewidths=1.3)
    axes.add_collection(bars)
    axes.plot(*pivot, marker="o", color="#334155", markersize=5)
    axes.set(xlim=(lower[0] - margin, upper[0] + margin),
             ylim=(lower[1] - margin, upper[1] + margin))
    axes.set_aspect("equal")
    axes.set_axis_off()
    return figure, bars


def update_chain(points: Point, edges: np.ndarray, bars: LineCollection) -> None:
    positions = points[0, 1:].dual().cast(ga.subspace("x y")).kernel
    bars.set_segments(positions[:, edges].reshape(-1, 2, 2))


def swinging_chain(frames: Iterable[Point], edges: np.ndarray, reference: Point) -> list[np.ndarray]:
    frames = tuple(frames)
    figure, bars = chain_scene(edges, reference, stack(frames))
    images = []
    for points in frames:
        update_chain(points, edges, bars)
        images.append(capture(figure))
    plt.close(figure)
    return images


def notebook_chain(frames: Iterable[Point], edges: np.ndarray, reference: Point, duration_ms: int) -> Path:
    images = swinging_chain(frames, edges, reference)
    return save_animation(images, "modal_xpbd_swing", duration_ms)


def draw_chain(points: Point, edges: np.ndarray) -> plt.Figure:
    """The chain at one moment: the bars of every moving girder, and the pin it hangs from."""
    figure, bars = chain_scene(edges, points, points[None])
    update_chain(points, edges, bars)
    return figure


def beam(frames: list[Point], edges: np.ndarray) -> list[np.ndarray]:
    """Frames of the crushed beam `[frames] [cases, bodies, vertices]`, the first case: its moving
    girders coloured in turn and its clamps grey, the whole span in a strip."""
    positions = stack(frames)[:, 0].dual().cast(ga.subspace("x y")).kernel   # [frames, bodies, vertices, 2]
    bodies = positions.shape[1]
    colours = np.where((np.arange(bodies) == 0) | (np.arange(bodies) == bodies - 1), "#94a3b8",
                       np.asarray(COLOURS)[np.arange(bodies) % len(COLOURS)])
    lower, upper = positions.min(axis=(0, 1, 2)), positions.max(axis=(0, 1, 2))
    reach = max(np.abs(positions[..., 1]).max(), (upper[0] - lower[0]) / 40)
    figure, axes = plt.subplots(figsize=(10, 1.6), dpi=60, layout="constrained")
    bars = LineCollection([], colors=np.repeat(colours, len(edges)), linewidths=0.8)
    axes.add_collection(bars)
    axes.set(xlim=(lower[0], upper[0]), ylim=(-1.2 * reach, 1.2 * reach))
    axes.set_aspect("equal")
    axes.set_axis_off()
    images = []
    for frame in positions:
        bars.set_segments(frame[:, edges].reshape(-1, 2, 2))
        images.append(capture(figure))
    plt.close(figure)
    return images


def buckling(crushing: np.ndarray, midspans: Point, critical: float) -> plt.Figure:
    """The beam's midspan deflection against how far its end was driven in, the first case, beside
    the Euler estimate."""
    deflection = np.abs(midspans[:, 0].dual().cast(ga.subspace("x y")).kernel[:, 1])
    figure, axes = plt.subplots(figsize=(6.5, 3), layout="constrained")
    axes.plot(crushing, deflection, color="#0f172a")
    axes.axvline(critical, linestyle=":", color="#dc2626", label="Euler estimate")
    axes.set(xlabel="end displacement", ylabel="midspan deflection")
    axes.spines[["top", "right"]].set_visible(False)
    axes.legend(frameon=False)
    return figure
