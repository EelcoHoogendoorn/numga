"""Drawing and coordinate readout for the Gaussian fit."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from examples.quadrics.gaussian.core import Point, Scalar, ga


def euclidean(points: Point) -> np.ndarray:
    coordinates = points.cast(ga.subspace("yw wx xy")).kernel
    return coordinates[..., :2] / coordinates[..., 2:]


def draw_gaussian(points: Point, pixels: Point, density: Scalar, level: Scalar) -> plt.Figure:
    xy, sample_xy = euclidean(pixels), euclidean(points)
    fig, ax = plt.subplots(figsize=(8, 7), dpi=140, layout="constrained")
    field = ax.pcolormesh(xy[..., 0], xy[..., 1], density.to_array(),
                          shading="auto", cmap="Blues", vmin=0, vmax=1, rasterized=True)
    ax.scatter(sample_xy[:, 0], sample_xy[:, 1], s=10, c="#26364a", alpha=0.4,
               linewidths=0, label="Point cloud")
    ax.contour(xy[..., 0], xy[..., 1], level.to_array(), levels=[0],
               colors=["#e56b24"], linewidths=2.2)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], color="#e56b24", lw=2.2))
    ax.legend(handles, labels + ["1σ quadric: Q(p) & p = 0"], loc="upper right")
    ax.set(xlabel="x", ylabel="y", aspect="equal",
           xlim=(xy[0, 0, 0], xy[0, -1, 0]), ylim=(xy[0, 0, 1], xy[-1, 0, 1]),
           title="A Gaussian and its 1σ quadric from point moments")
    fig.colorbar(field, ax=ax, shrink=0.72, label="Density / peak density")
    return fig
