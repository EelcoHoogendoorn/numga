"""Drawing for the skinning example: each skin as a striped quad mesh."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from numga import stack
from examples.surfaces.skinning.core import Point, mv


# --- plumbing -------------------------------------------------------------------------
def xyz(points: Point) -> np.ndarray:
    """Coordinates of points: their pairings with the planes x, y and z, over their weights."""
    return ((stack((mv.x, mv.y, mv.z)) & points[..., None]) / (mv.w & points)[..., None]).to_array()


def surface(ax, points: Point, n_rings: int, n_around: int) -> None:
    """Draw the skinned cylinder as a shaded quad mesh, striped along its length."""
    k = xyz(points).reshape(n_rings, n_around, 3)
    # Close each ring:
    k = np.concatenate([k, k[:, :1]], axis=1)
    stripes = plt.get_cmap("viridis")(np.linspace(0.0, 1.0, n_around + 1))[None].repeat(n_rings, axis=0)
    ax.plot_surface(k[..., 0], k[..., 1], k[..., 2], facecolors=stripes, edgecolor="black", linewidth=0.2, shade=True)
    ax.set_box_aspect((1, 1, 1)); ax.set_xlim(0, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)


def draw_skinning(motor_skin: Point, slerp_skin: Point, matrix_skin: Point, n_rings: int, n_around: int) -> plt.Figure:
    """The three skins side by side: blended motors normalised, slerped, and blended maps."""
    fig = plt.figure(figsize=(15, 5), dpi=120)
    for index, skin in enumerate((motor_skin, slerp_skin, matrix_skin), start=1):
        surface(fig.add_subplot(1, 3, index, projection="3d"), skin, n_rings, n_around)
    return fig
