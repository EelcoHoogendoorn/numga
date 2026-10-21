"""Drawing for the Dirac electron: the mass shell, and trembling paths with their spin."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from numga import stack
from examples.animation import capture
from examples.relativity.dirac import core

colours = ("#2e86c1", "#7d3c98", "#c0392b")


def spatial(directions: core.Antivector) -> np.ndarray:
    """Directions as the observer sees them: read by pairing with the slices x, y and z, on the last axis."""
    mv = directions.context.multivector
    return (stack((mv.x, mv.y, mv.z)) & directions[..., None]).to_array()


def draw_paths(times: np.ndarray, spinors: list, paths: list, mixtures: tuple) -> plt.Figure:
    """The paths the currents trace, one per share of negative energy, with the spin at the start."""
    figure = plt.figure(figsize=(7, 7))
    ax = figure.add_subplot(projection="3d")
    draw_scene(ax, spinors, paths, mixtures, len(times))
    figure.tight_layout()
    return figure


def draw_scene(ax, spinors: list, paths: list, mixtures: tuple, upto: int) -> None:
    for psi, trace, share, colour in zip(spinors, paths, mixtures, colours):
        xyz = spatial(trace)[:upto]                                           # [times, 3]
        ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=colour, linewidth=1.2, label=f"{share:.0%} negative energy")
        here = xyz[-1]
        axis = spatial(core.spin(psi[upto - 1]))
        axis = 0.4 * axis / np.linalg.norm(axis)
        ax.quiver(*here, *axis, color=colour, linewidth=1.5, arrow_length_ratio=0.25)
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(-0.6, 0.6)
    ax.set_zlim(-0.2, 3.2)
    ax.set_box_aspect((1.2, 1.2, 3.4))
    ax.set_axis_off()
    ax.legend(loc="upper left", fontsize=8)
    ax.view_init(elev=12, azim=-60)


def animate_paths(times: np.ndarray, spinors: list, paths: list, mixtures: tuple, every: int) -> list[np.ndarray]:
    """The paths drawn as they are traced, with each electron's spin at its current position."""
    frames = []
    for upto in range(every, len(times) + 1, every):
        figure = plt.figure(figsize=(5, 6))
        ax = figure.add_subplot(projection="3d")
        draw_scene(ax, spinors, paths, mixtures, upto)
        figure.tight_layout()
        frames.append(capture(figure))
        plt.close(figure)
    return frames
