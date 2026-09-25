"""Drawing for the pose graph: the true, dead-reckoned and most likely paths, with each pose's ellipse
drawn as the zero level of its quadric on a grid of points. Lengths are in metres."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.animation import capture
from examples.geometry.belief_propagation import core


def point(coords: np.ndarray) -> core.Point:
    """Points of unit weight at (..., 2) coordinates: the dual of the homogeneous vector."""
    return (core.mv("x y", coords) + core.mv.w).dual()


def xy(points: core.Point) -> np.ndarray:
    """Euclidean coordinates of points: their pairings with the coordinate lines, at unit weight."""
    return np.stack([((line & points) / (core.mv.w & points)).to_array() for line in (core.mv.x, core.mv.y)], axis=-1)


def extent(*paths: core.Motor) -> np.ndarray:
    """[2, 2] the lower and upper corners of a box around the origins the paths carry, with a margin."""
    corners = np.concatenate([xy(path >> core.ORIGIN).reshape(-1, 2) for path in paths])
    low, high = corners.min(axis=0), corners.max(axis=0)
    margin = 0.15 * (high - low).max()
    return np.stack([low - margin, high + margin])


def frame(ax, box: np.ndarray, title: str) -> None:
    """Equal axes over the box, without ticks."""
    ax.set_xlim(box[:, 0])
    ax.set_ylim(box[:, 1])
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def draw_paths(ax, truth: core.Motor, dead: core.Motor) -> None:
    """The true and the dead-reckoned path."""
    ax.plot(*xy(truth >> core.ORIGIN).T, color="0.6", linewidth=2.5, label="truth")
    ax.plot(*xy(dead >> core.ORIGIN).T, "--", color="#c0392b", linewidth=1.2, label="dead reckoning")


def draw_lap(truth: core.Motor, dead: core.Motor, linked: core.Motor) -> plt.Figure:
    """The true lap and dead reckoning, with the two dead-reckoned poses a reading links."""
    figure, ax = plt.subplots(figsize=(5.2, 5.2))
    draw_paths(ax, truth, dead)
    ax.plot(*xy(linked >> core.ORIGIN).T, "o-", color="#2a9d4a", markersize=5, linewidth=1.5, label="closing reading")
    frame(ax, extent(truth, dead), "the lap")
    ax.legend(loc="lower left", fontsize=8)
    figure.tight_layout()
    return figure


def draw_panel(ax, truth: core.Motor, dead: core.Motor, poses: core.Motor, ellipses: core.Quadric,
               box: np.ndarray, title: str) -> None:
    """The true and dead-reckoned paths, and the poses with their ellipses."""
    x, y = np.meshgrid(*np.linspace(box[0], box[1], 240).T)
    grid = point(np.stack([x, y], axis=-1))                                   # [rows, columns] Point
    levels = (ellipses[:, None, None](grid) & grid).to_array()                # [poses, rows, columns]
    draw_paths(ax, truth, dead)
    ax.plot(*xy(poses >> core.ORIGIN).T, "o-", color="#2c5d9e", markersize=2.5, linewidth=1.0, label="belief")
    for level in levels:
        ax.contour(x, y, level, levels=[0.0], colors="#2c5d9e", linewidths=0.8, alpha=0.7)
    frame(ax, box, title)


def draw_survey(runs: dict, index: int = -1) -> plt.Figure:
    """Each named run in a panel of its own, after the given round: the true and dead-reckoned paths,
    and the poses with their ellipses."""
    truth, dead, poses, _ = next(iter(runs.values()))
    box = extent(truth, dead)
    figure, axes = plt.subplots(1, len(runs), figsize=(5 * len(runs), 5.2), squeeze=False)
    for ax, (name, (truth, dead, poses, ellipses)) in zip(axes[0], runs.items()):
        draw_panel(ax, truth, dead, poses[index], ellipses[index], box, name)
    figure.suptitle(f"after {index % len(poses) + 1} rounds of belief propagation")
    axes[0, 0].legend(loc="lower left", fontsize=8)
    figure.tight_layout()
    return figure


def animate_survey(runs: dict, frames: int) -> list[np.ndarray]:
    """Frames at rounds spaced evenly in their logarithm, from the first round to the last."""
    _, _, poses, _ = next(iter(runs.values()))
    images = []
    for index in np.unique(np.geomspace(1, len(poses), frames).astype(int)) - 1:
        figure = draw_survey(runs, index)
        images.append(capture(figure))
        plt.close(figure)
    return images
