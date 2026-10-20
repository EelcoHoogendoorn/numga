"""Drawing and coordinate readout for the Gaussian fit."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from numga import stack
from examples.quadrics.gaussian.core import Point, Scalar, mv


# --- plumbing -------------------------------------------------------------------------
def euclidean(points: Point) -> np.ndarray:
    """Coordinates of points, `[..., 2]`: their pairings with the lines x = 0 and y = 0, over their weights."""
    return ((stack((mv.x, mv.y)) & points[..., None]) / (mv.w & points)[..., None]).to_array()


def draw_gaussian(points: Point, pixels: Point, density: Scalar, level: Scalar) -> plt.Figure:
    """The fitted density over the pixels, the point cloud, and the 1σ quadric, orange."""
    xy, sample_xy = euclidean(pixels), euclidean(points)
    fig, ax = plt.subplots(figsize=(7, 7), dpi=140, layout="constrained")
    ax.pcolormesh(xy[..., 0], xy[..., 1], density.to_array(), shading="auto", cmap="Blues", vmin=0, vmax=1, rasterized=True)
    ax.scatter(sample_xy[:, 0], sample_xy[:, 1], s=10, c="#26364a", alpha=0.4, linewidths=0)
    ax.contour(xy[..., 0], xy[..., 1], level.to_array(), levels=[0], colors=["#e56b24"], linewidths=2.2)
    ax.set(xlabel="x", ylabel="y", aspect="equal", xlim=(xy[0, 0, 0], xy[0, -1, 0]), ylim=(xy[0, 0, 1], xy[-1, 0, 1]))
    return fig
