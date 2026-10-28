"""Drawing and coordinate readout for the Gaussian fit."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from numga import stack
from collections.abc import Callable

from examples.quadrics.gaussian.core import Point, Polarity, Scalar, mv
from examples.util.wgsl import pinned, room

# The box the fit is shown over, along x and y, and the figure's width in inches, at 100 pixels an inch.
BOX = np.array([[-5.0, -5.0], [6.0, 4.0]])
WIDTH = 3.0


# --- plumbing -------------------------------------------------------------------------
def euclidean(points: Point) -> np.ndarray:
    """Coordinates of points, `[..., 2]`: their pairings with the planes x = 0 and y = 0, over their weights."""
    return ((stack((mv.x, mv.y)) & points[..., None]) / (mv.w & points)[..., None]).to_array()


def draw_gaussian(points: Point, density: Callable[[Point], Scalar], ellipsoid: Polarity) -> plt.Figure:
    """The density over the pixels, the point cloud, and the 1σ ellipsoid, orange, the zero contour of its form."""
    (left, bottom), (right, top) = BOX
    fig = plt.figure(figsize=(WIDTH, WIDTH * (top - bottom) / (right - left)), dpi=100)
    ax = fig.add_axes((0, 0, 1, 1))
    columns, rows = room(ax)
    # One point at the middle of each pixel, rows down from the top.
    across = left + (np.arange(columns) + 0.5) * (right - left) / columns
    down = top - (np.arange(rows) + 0.5) * (top - bottom) / rows
    pixels = mv.point(np.stack(np.meshgrid(across, down), axis=-1))             # [rows, columns] Point-
    shades = plt.get_cmap("Blues")(np.clip(density(pixels).to_array(), 0, 1))
    pinned(ax, np.round(shades[..., :3] * 255).astype(np.uint8), (left, right, bottom, top))
    sample_xy = euclidean(points)
    ax.scatter(sample_xy[:, 0], sample_xy[:, 1], s=4, c="#26364a", alpha=0.4, linewidths=0)
    ax.contour(across, down, (ellipsoid(pixels) & pixels).to_array(), levels=[0], colors=["#e56b24"], linewidths=1.5)
    ax.set(xlim=(left, right), ylim=(bottom, top))
    cross(ax, "x", "y")
    return fig


def cross(ax, across: str, up: str) -> None:
    """No scales: a faint cross through the origin over the density, the coordinate planes through it, each named at its
    end."""
    ax.axhline(0, color="0.8", linewidth=0.7, zorder=1.5)
    ax.axvline(0, color="0.8", linewidth=0.7, zorder=1.5)
    ax.annotate(across, (1, 0), xycoords=("axes fraction", "data"), xytext=(-4, -3), textcoords="offset points",
                ha="right", va="top", fontsize=8, color="0.4")
    ax.annotate(up, (0, 1), xycoords=("data", "axes fraction"), xytext=(4, -4), textcoords="offset points",
                ha="left", va="top", fontsize=8, color="0.4")
    ax.set_axis_off()
