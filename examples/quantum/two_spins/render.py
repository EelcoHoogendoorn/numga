"""Drawing for two spins: each spin's Bloch vector in its ball, the correlation ellipsoid between them,
the image of the second spin's unit directions under the correlation map, and the largest Bell
combination over the exchange."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.animation import capture
from examples.quantum.two_spins import core

FIRST, SECOND, TOGETHER = "#c0392b", "#2e86c1", "#7d3c98"


def components(vectors: core.First | core.Second, names: str) -> np.ndarray:
    """The coordinates of vectors along the named directions."""
    return vectors.cast(vectors.algebra.subspace(names)).kernel


def ball(ax) -> None:
    """The unit sphere, faintly, with its axes."""
    u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 40), np.linspace(0, np.pi, 20))
    ax.plot_wireframe(np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v), color="0.85", linewidth=0.4)
    for axis in np.eye(3):
        ax.plot(*np.stack([-axis, axis]).T, color="0.6", linewidth=0.6)
    ax.set(xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1))
    ax.set_box_aspect((1, 1, 1), zoom=1.4)
    ax.set_axis_off()
    ax.view_init(elev=18, azim=-55)


def arrow(ax, tip: np.ndarray, colour: str) -> None:
    ax.plot(*np.stack([np.zeros(3), tip]).T, color=colour, linewidth=2.5)
    ax.scatter(*tip, color=colour, s=25)


def ellipsoid(ax, correlations: core.Correlation) -> None:
    """The image of the second spin's unit directions under the correlation map."""
    u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 36), np.linspace(0, np.pi, 18))
    directions = correlations.context.multivector(correlations.input_subspaces[0],
                                                  np.stack([np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)], axis=-1))
    surface = components(correlations(directions), "x y z")                   # [rows, columns, 3]
    ax.plot_surface(*np.moveaxis(surface, -1, 0), color=TOGETHER, alpha=0.35, linewidth=0.3, edgecolor=TOGETHER)


def draw_panels(figure: plt.Figure, first: core.First, second: core.Second, correlations: core.Correlation,
                bottom: float) -> None:
    """Each spin's Bloch vector in its ball, and the correlation ellipsoid between them, above the
    given height of the figure."""
    for column, title in enumerate(("first spin", "correlations", "second spin")):
        ax = figure.add_axes((column / 3, bottom, 1 / 3, 0.9 - bottom), projection="3d")
        ball(ax)
        figure.text((column + 0.5) / 3, 0.94, title, ha="center", fontsize=12)
        if column == 0:
            arrow(ax, components(first, "x y z"), FIRST)
        elif column == 1:
            ellipsoid(ax, correlations)
        else:
            arrow(ax, components(second, "X Y Z"), SECOND)


def draw_state(first: core.First, second: core.Second, correlations: core.Correlation) -> plt.Figure:
    """The two Bloch vectors and the correlation ellipsoid of one state of the pair."""
    figure = plt.figure(figsize=(12, 4.4))
    draw_panels(figure, first, second, correlations, 0.0)
    return figure


def draw_pair(angles: np.ndarray, first: core.First, second: core.Second, correlations: core.Correlation,
              bells: core.Scalar, index: int) -> plt.Figure:
    """The two Bloch vectors and the correlation ellipsoid at one angle of the exchange, over the
    largest Bell combination along the whole exchange."""
    figure = plt.figure(figsize=(12, 6))
    draw_panels(figure, first[index], second[index], correlations[index], 0.3)
    trace = figure.add_axes((0.08, 0.08, 0.86, 0.18))
    values = bells.to_array()
    trace.plot(angles, values, color=TOGETHER, linewidth=1.5)
    trace.scatter(angles[index], values[index], color=TOGETHER, s=30, zorder=3)
    trace.axhline(2.0, color="0.5", linestyle="--", linewidth=0.8)
    trace.axhline(2 * np.sqrt(2), color="0.5", linestyle=":", linewidth=0.8)
    trace.text(angles[len(angles) // 2], 2.0, "each spin its own answers", ha="center", va="bottom", fontsize=8, color="0.4")
    trace.text(angles[0], 2 * np.sqrt(2), " 2√2", ha="left", va="bottom", fontsize=8, color="0.4")
    trace.set(xlim=(angles[0], angles[-1]), ylim=(1.9, 2.95), xlabel="exchange angle (rad)", ylabel="largest Bell value")
    return figure


def animate_pair(angles: np.ndarray, first: core.First, second: core.Second, correlations: core.Correlation,
                 bells: core.Scalar) -> list[np.ndarray]:
    """One frame per angle of the exchange."""
    frames = []
    for index in range(len(angles)):
        figure = draw_pair(angles, first, second, correlations, bells, index)
        frames.append(capture(figure))
        plt.close(figure)
    return frames
