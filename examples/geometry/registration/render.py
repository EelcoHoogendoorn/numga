"""Drawing for the registration example: read the points out as coordinates and plot them."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from numga import stack
from examples.geometry.registration.core import Point, mv

# The margin around the points, as a fraction of their largest extent.
MARGIN = 0.08


def coordinates(points: Point) -> np.ndarray:
    """Coordinates of points: their pairings with the planes x, y and z, over their weights."""
    return ((stack((mv.x, mv.y, mv.z)) & points[..., None]) / (mv.w & points)[..., None]).to_array()


def draw_registration(source: Point, target: Point, aligned: Point) -> plt.Figure:
    """The source, grey, the target, blue crosses, and the aligned source, red, each aligned point tied to
    its target."""
    fig = plt.figure(figsize=(8, 7), dpi=120, layout="constrained")
    ax = fig.add_subplot(projection="3d")
    source_xyz, target_xyz, aligned_xyz = map(coordinates, (source, target, aligned))
    ax.scatter(*source_xyz.T, color="#94a3b8", s=10)
    ax.scatter(*target_xyz.T, color="#0284c7", s=18, marker="x")
    ax.scatter(*aligned_xyz.T, color="#f43f5e", s=10)
    for p, q in zip(aligned_xyz, target_xyz):
        ax.plot(*np.stack([p, q]).T, color="#f43f5e", linewidth=0.5, alpha=0.6)
    # Framed by everything drawn, at equal aspect.
    drawn = np.concatenate([source_xyz, target_xyz, aligned_xyz])
    low, high = drawn.min(axis=0), drawn.max(axis=0)
    pad = MARGIN * (high - low).max()
    ax.set(xlim=(low[0] - pad, high[0] + pad), ylim=(low[1] - pad, high[1] + pad), zlim=(low[2] - pad, high[2] + pad))
    ax.set_box_aspect(high - low + 2 * pad)
    ax.set_axis_off()
    return fig
