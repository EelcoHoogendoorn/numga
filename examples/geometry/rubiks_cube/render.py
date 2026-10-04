"""Drawing for the cube: each sticker a square, coloured by the face it belongs to."""

from __future__ import annotations

from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from examples.animation import capture
from examples.geometry.rubiks_cube import core

# The faces' colours, by side and axis: +x red, +y blue, +z white, -x orange, -y green, -z yellow; and
# the view.
COLOURS = np.array(["#c41e3a", "#0051ba", "#ffffff", "#ff5800", "#009e60", "#ffd500"])
ELEVATION, AZIMUTH = 28.0, -58.0


def coordinates(points: core.Vector) -> np.ndarray:
    """The x, y and z coordinates of points."""
    return points.cast(core.ga.subspace("x y z")).kernel


def draw(ax, corners: core.Vector, colours: np.ndarray, cut: core.Vector) -> None:
    """The cube's stickers as squares coloured by their faces, with dark edges, and the dark cut under
    the turning layer on both sides; in cubies about the cube's centre."""
    squares = np.concatenate([coordinates(corners), coordinates(cut).reshape(-1, 4, 3)]) / 2
    fills = np.concatenate([COLOURS[colours], np.full(cut.shape[0] * cut.shape[1], "#111111")])
    ax.add_collection3d(Poly3DCollection(squares, facecolors=fills, edgecolors="#111111", linewidths=1.2))
    for set_limit in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
        set_limit(-1.6, 1.6)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=ELEVATION, azim=AZIMUTH)
    ax.set_axis_off()


def animate(states: Iterable[tuple[core.Vector, core.Vector]], colours: np.ndarray) -> list[np.ndarray]:
    """A frame of the cube for each state of its stickers' corners and its turning cut."""
    images = []
    for corners, cut in states:
        figure = plt.figure(figsize=(4, 4), dpi=90)
        draw(figure.add_subplot(projection="3d"), corners, colours, cut)
        images.append(capture(figure))
        plt.close(figure)
    return images
