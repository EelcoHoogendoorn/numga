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
LIMITS = np.array([[-3.3, 3.5], [-1.5, 1.5], [-1.1, 4.0]])


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


def side_by_side(bar, shapes: list[core.Vector], titles: list[str], cell: np.ndarray) -> plt.Figure:
    """The bar in each of the given shapes, side by side, checkered by where each triangle sat at rest."""
    colours = checker(bar.vertices, bar.faces, cell)
    figure = plt.figure(figsize=(4 * len(shapes), 4), dpi=80)
    for i, (shape, title) in enumerate(zip(shapes, titles)):
        draw(figure.add_subplot(1, len(shapes), i + 1, projection="3d"), shape, bar.faces, colours, title)
    return figure


def animate(bar, shapes: core.Vector, cell: np.ndarray) -> list[np.ndarray]:
    """The bar in each of the given shapes, one frame each."""
    images = []
    colours = checker(bar.vertices, bar.faces, cell)
    for shape in shapes:
        figure = plt.figure(figsize=(4, 4), dpi=80)
        draw(figure.add_axes((0, 0, 1, 1), projection="3d"), shape, bar.faces, colours, "")
        images.append(capture(figure))
        plt.close(figure)
    return images


def inline(frames: list[np.ndarray], duration_ms: int):
    """Frames as a looping GIF to show in a notebook, kept in memory."""
    from io import BytesIO
    from IPython.display import Image as Shown
    from PIL import Image
    images = [Image.fromarray(pixels) for pixels in frames]
    buffer = BytesIO()
    images[0].save(buffer, format="GIF", save_all=True, append_images=images[1:], duration=duration_ms, loop=0)
    return Shown(data=buffer.getvalue(), format="gif")
