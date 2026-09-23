"""Drawing for the skinning example: each skin as a striped quad mesh."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.geometry.skinning.core import Point, ga


def xyz(points: Point) -> np.ndarray:
    k = points.cast(ga.subspace("yzw zxw xyw zyx")).kernel
    return k[..., :3] / k[..., 3:]


def surface(ax, points: Point, rings: int, around: int, title: str) -> None:
    """Draw the skinned cylinder as a shaded quad mesh, striped along its length."""
    k = xyz(points).reshape(rings, around, 3)
    k = np.concatenate([k, k[:, :1]], axis=1)                       # close each ring
    stripes = plt.get_cmap("viridis")(np.linspace(0.0, 1.0, around + 1))[None].repeat(rings, axis=0)
    ax.plot_surface(k[..., 0], k[..., 1], k[..., 2], facecolors=stripes, edgecolor="black", linewidth=0.2, shade=True)
    ax.set_title(title); ax.set_box_aspect((1, 1, 1)); ax.set_xlim(0, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)


def draw_skinning(motor_skin: Point, slerp_skin: Point, matrix_skin: Point, rings: int, around: int) -> plt.Figure:
    """The three skins side by side."""
    fig = plt.figure(figsize=(15, 5), dpi=120)
    surface(fig.add_subplot(1, 3, 1, projection="3d"), motor_skin, rings, around, "motor blend (normalised lerp)")
    surface(fig.add_subplot(1, 3, 2, projection="3d"), slerp_skin, rings, around, "motor blend (slerp)")
    surface(fig.add_subplot(1, 3, 3, projection="3d"), matrix_skin, rings, around, "matrix blend")
    return fig
