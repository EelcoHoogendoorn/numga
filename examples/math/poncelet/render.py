"""Drawing for Poncelet's porism: the two pairs of conics side by side, each conic the zero level of
its pairing with a grid of points, and the path of five sides from the same start about each inner
conic."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.animation import capture
from examples.math.poncelet import core

# The view: x from -1.8 to 2.6, y from -1.9 to 1.5.
BOX = np.array([[-1.8, -1.9], [2.6, 1.5]])
TITLES = ("closes from every start", "misses from every start")


def xy(points: core.Point) -> np.ndarray:
    """Euclidean coordinates of points: their pairings with the coordinate lines, at unit weight."""
    return np.stack([((line & points) / (core.mv.w & points)).to_array() for line in (core.mv.x, core.mv.y)], axis=-1)


def draw(outer: core.Conic, inner: core.Conic, vertices: core.Point) -> plt.Figure:
    """For each inner conic, the ellipse, the inner conic and the path of sides about it from the same
    start, the start orange and the last vertex ringed."""
    x, y = np.meshgrid(*np.linspace(BOX[0], BOX[1], 300).T)
    grid = core.point(np.stack([x, y], axis=-1))                              # [rows, columns] Point
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    corners = xy(vertices)                                                     # [steps + 1, inners, 2]
    for index, ax in enumerate(axes):
        ax.contour(x, y, (outer(grid) & grid).to_array(), levels=[0.0], colors="0.25", linewidths=1.6)
        ax.contour(x, y, (inner[index](grid) & grid).to_array(), levels=[0.0], colors="#2c5d9e", linewidths=1.4)
        ax.plot(*corners[:, index].T, color="#c0392b", linewidth=1.4)
        ax.scatter(*corners[0, index], color="#e67e22", s=50, zorder=3)
        ax.scatter(*corners[-1, index], facecolors="none", edgecolors="#c0392b", s=110, linewidths=1.5, zorder=3)
        ax.set(xlim=BOX[:, 0], ylim=BOX[:, 1], aspect="equal", xticks=[], yticks=[], title=TITLES[index])
    figure.tight_layout()
    return figure


def animate(scenes: list[tuple[core.Conic, core.Conic, core.Point]]) -> list[np.ndarray]:
    """One frame per scene."""
    frames = []
    for scene in scenes:
        figure = draw(*scene)
        frames.append(capture(figure))
        plt.close(figure)
    return frames
