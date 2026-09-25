"""Drawing for Pascal's theorem: the conic, the sides of the hexagon extended across the view, the
crossings of opposite sides and the line through them, each curve the zero level of its pairing with
a grid of points."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.animation import capture
from examples.math.pascal import core

# The view: x from -2.5 to 4.5, y from -2.5 to 5.5.
BOX = np.array([[-2.5, -2.5], [4.5, 5.5]])
# Colours of the three pairs of opposite sides and their crossings.
PAIRS = np.array(["#1f77b4", "#2a9d4a", "#9b59b6"])


def xy(points: core.Point) -> np.ndarray:
    """Euclidean coordinates of points: their pairings with the coordinate lines, at unit weight."""
    return np.stack([((line & points) / (core.mv.w & points)).to_array() for line in (core.mv.x, core.mv.y)], axis=-1)


def draw_on(ax: plt.Axes, shape: core.Conic, hexagon: core.Point, crossing: core.Point) -> None:
    x, y = np.meshgrid(*np.linspace(BOX[0], BOX[1], 400).T)
    grid = core.point(np.stack([x, y], axis=-1))                              # [rows, columns] Point
    sides = hexagon & hexagon[core.NEXT]                                       # [6] Plane
    pascal_line = crossing[0] & crossing[2]                                    # [] Plane
    ax.contour(x, y, (shape(grid) & grid).to_array(), levels=[0.0], colors="0.25", linewidths=1.6)
    for side, colour in zip(sides, np.tile(PAIRS, 2)):
        ax.contour(x, y, (side & grid).to_array(), levels=[0.0], colors=colour, linewidths=0.8, alpha=0.6)
    ax.contour(x, y, (pascal_line & grid).to_array(), levels=[0.0], colors="#c0392b", linewidths=2.0)
    ax.fill(*xy(hexagon).T, color="0.85", alpha=0.5, zorder=0)
    ax.scatter(*xy(hexagon).T, color="0.15", s=28, zorder=3)
    ax.scatter(*xy(hexagon[-1:]).T, color="#e67e22", s=60, zorder=4)
    ax.scatter(*xy(crossing).T, color=PAIRS, s=55, edgecolors="#c0392b", linewidths=1.5, zorder=4)
    ax.set(xlim=BOX[:, 0], ylim=BOX[:, 1], aspect="equal", xticks=[], yticks=[])


def draw(shape: core.Conic, hexagon: core.Point, crossing: core.Point) -> plt.Figure:
    """Six points on a conic as a hexagon, its opposite sides in matching colours, and the three
    points where they meet on one line."""
    figure, ax = plt.subplots(figsize=(5.6, 6.2))
    draw_on(ax, shape, hexagon, crossing)
    figure.tight_layout()
    return figure


def animate(scenes: list[tuple[core.Conic, core.Point, core.Point]]) -> list[np.ndarray]:
    """One frame per scene."""
    frames = []
    for scene in scenes:
        figure = draw(*scene)
        frames.append(capture(figure))
        plt.close(figure)
    return frames
