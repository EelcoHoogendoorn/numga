"""Drawing for as rigid as possible: the bar's two deformations side by side, checkered by where each
triangle sat at rest."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from examples.animation import capture
from examples.surfaces.arap import core

# The checkerboard's two colours, the view, and the box it shows.
LIGHT, DARK = np.array([0.98, 0.84, 0.62]), np.array([0.90, 0.45, 0.18])
ELEVATION, AZIMUTH = 18.0, -64.0
LIMITS = np.array([[-3.4, 2.2], [-1.4, 1.4], [-1.2, 5.0]])


def coordinates(points: core.Vector) -> np.ndarray:
    """The x, y and z coordinates of points."""
    return points.cast(core.ga.subspace("x y z")).kernel


def checker(rest: core.Vector, faces: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Each triangle's colour: the parity of the cell of the given size its centroid sat in at rest."""
    centroids = coordinates(rest)[faces].mean(axis=1)                          # [F, 3]
    parity = np.floor(centroids / cell).astype(int).sum(axis=-1) % 2
    return np.where(parity[:, None] == 1, DARK, LIGHT)


def draw(ax, vertices: core.Vector, faces: np.ndarray, colours: np.ndarray, title: str) -> None:
    """The bar's triangles, shaded, in the fixed box."""
    ax.add_collection3d(Poly3DCollection(coordinates(vertices)[faces], facecolors=colours, shade=True, linewidths=0))
    for set_limit, limit in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), LIMITS):
        set_limit(*limit)
    ax.set_box_aspect(LIMITS[:, 1] - LIMITS[:, 0])
    ax.view_init(elev=ELEVATION, azim=AZIMUTH)
    ax.set_axis_off()
    ax.set_title(title)


def animate(bar, rigid: core.Vector, laplacian: core.Vector, cell: np.ndarray) -> list[np.ndarray]:
    """Each frame: the bar as rigid as possible beside Laplacian editing."""
    colours = checker(bar.vertices, bar.faces, cell)
    images = []
    for shape, edited in zip(rigid, laplacian):
        figure = plt.figure(figsize=(8, 4), dpi=80)
        draw(figure.add_subplot(1, 2, 1, projection="3d"), shape, bar.faces, colours, "as rigid as possible")
        draw(figure.add_subplot(1, 2, 2, projection="3d"), edited, bar.faces, colours, "Laplacian editing")
        images.append(capture(figure))
        plt.close(figure)
    return images
