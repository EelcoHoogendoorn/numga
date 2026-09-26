"""Drawing for the pose graph: the true, dead-reckoned and most likely paths, with each pose's ellipse
drawn as the zero level of its quadric on a grid of points. Lengths are in metres."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.animation import capture
from numga.algebras import PGA2D
from examples import instantiate

# The drawings are in the plane.
core = instantiate("examples.geometry.belief_propagation.core", PGA2D)
# The point each pose carries: the origin, dual to the weight direction w.
ORIGIN = core.mv.w.dual()                                                     # [] Point


def point(coords: np.ndarray) -> core.Point:
    """Points of unit weight at (..., 2) coordinates: the dual of the homogeneous vector."""
    return (core.mv("x y", coords) + core.mv.w).dual()


def xy(points: core.Point) -> np.ndarray:
    """Euclidean coordinates of points: their pairings with the coordinate lines, at unit weight."""
    return np.stack([((line & points) / (core.mv.w & points)).to_array() for line in (core.mv.x, core.mv.y)], axis=-1)


def extent(*paths: core.Motor) -> np.ndarray:
    """[2, 2] the lower and upper corners of a box around the origins the paths carry, with a margin."""
    corners = np.concatenate([xy(path >> ORIGIN).reshape(-1, 2) for path in paths])
    low, high = corners.min(axis=0), corners.max(axis=0)
    margin = 0.15 * (high - low).max()
    return np.stack([low - margin, high + margin])


def frame(ax, box: np.ndarray) -> None:
    """Equal axes over the box, without frame or ticks."""
    ax.set_xlim(box[:, 0])
    ax.set_ylim(box[:, 1])
    ax.set_aspect("equal")
    ax.set_axis_off()


def draw_paths(ax, truth: core.Motor, dead: core.Motor) -> None:
    """The true and the dead-reckoned path."""
    ax.plot(*xy(truth >> ORIGIN).T, color="0.75", linewidth=3.0, label="truth")
    ax.plot(*xy(dead >> ORIGIN).T, "--", color="#d9822b", linewidth=1.4, label="dead reckoning")
    # The known start, where both paths begin.
    ax.plot(*xy(truth[:1] >> ORIGIN).T, "o", color="0.2", markersize=7, zorder=5)


def draw_lap(truth: core.Motor, dead: core.Motor, linked: core.Motor) -> plt.Figure:
    """The true lap and dead reckoning, with the two dead-reckoned poses a reading links."""
    figure, ax = plt.subplots(figsize=(5.2, 5.2))
    draw_paths(ax, truth, dead)
    ax.plot(*xy(linked >> ORIGIN).T, "o-", color="#2a9d4a", markersize=5, linewidth=1.5, label="closing reading")
    frame(ax, extent(truth, dead))
    ax.legend(loc="lower left", fontsize=8, frameon=False)
    figure.tight_layout()
    return figure


def draw_panel(ax, truth: core.Motor, dead: core.Motor, poses: core.Motor, ellipses: core.Quadric,
               box: np.ndarray) -> None:
    """The true and dead-reckoned paths, and the poses with their ellipses."""
    x, y = np.meshgrid(*np.linspace(box[0], box[1], 240).T)
    grid = point(np.stack([x, y], axis=-1))                                   # [rows, columns] Point
    levels = (ellipses[:, None, None](grid) & grid).to_array()                # [poses, rows, columns]
    draw_paths(ax, truth, dead)
    for level in levels:
        ax.contour(x, y, level, levels=[0.0], colors="#2c5d9e", linewidths=0.8, alpha=0.7)
    ax.plot(*xy(poses >> ORIGIN).T, "o-", color="#2c5d9e", markersize=2.5, linewidth=1.0, label="belief")
    frame(ax, box)


def draw_survey(runs: dict, index: int = -1) -> plt.Figure:
    """Each run in a panel of its own, after the given round: the true and dead-reckoned paths, and
    the poses with their ellipses."""
    truth, dead, poses, _ = next(iter(runs.values()))
    box = extent(truth, dead)
    figure, axes = plt.subplots(1, len(runs), figsize=(5 * len(runs), 5), squeeze=False)
    for ax, (truth, dead, poses, ellipses) in zip(axes[0], runs.values()):
        draw_panel(ax, truth, dead, poses[index], ellipses[index], box)
    axes[0, 0].legend(loc="lower left", fontsize=8, frameon=False)
    figure.tight_layout()
    return figure


def log_rounds(rounds: int, frames: int) -> np.ndarray:
    """Round indices spaced evenly in their logarithm, from the first round to the last."""
    return np.unique(np.geomspace(1, rounds, frames).astype(int)) - 1


def animate_survey(runs: dict, rounds: np.ndarray) -> list[np.ndarray]:
    """One frame after each of the given rounds."""
    images = []
    for index in rounds:
        figure = draw_survey(runs, index)
        images.append(capture(figure))
        plt.close(figure)
    return images


def evenly_moving(poses: core.Motor, frames: int) -> np.ndarray:
    """Round indices at which the poses have made equal shares of their whole motion, measured by the
    largest move of any carried origin in each round: the frames of an animation that moves evenly."""
    moved = np.abs(np.diff(xy(poses >> ORIGIN), axis=0)).max(axis=(-2, -1))   # [rounds - 1]
    share = np.cumsum(moved) / moved.sum()
    return np.unique(np.searchsorted(share, np.linspace(0, 1, frames + 1)[1:] - 1e-9)) + 1


def animate_growth(truth: core.Motor, dead: core.Motor, frames) -> list[np.ndarray]:
    """One image for each frame of shown poses, the poses and their ellipses, over the paths as far as
    the shown poses reach, on one box for the whole lap."""
    box = extent(truth, dead)
    images = []
    for shown, poses, ellipses in frames:
        figure, ax = plt.subplots(figsize=(5, 5))
        draw_panel(ax, truth[:shown], dead[:shown], poses, ellipses, box)
        ax.legend(loc="lower left", fontsize=8, frameon=False)
        figure.tight_layout()
        images.append(capture(figure))
        plt.close(figure)
    return images

