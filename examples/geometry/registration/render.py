"""Drawing for the registration example: read the points out as coordinates and plot them."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.geometry.registration.core import Point, ga


def coordinates(points: Point) -> np.ndarray:
    values = points.cast(ga.subspace("yzw zxw xyw zyx")).kernel
    return values[..., :3] / values[..., 3:]


def draw_registration(source: Point, target: Point, aligned: Point, title: str) -> plt.Figure:
    """Source, target and aligned source, with each aligned point tied to its target."""
    fig = plt.figure(figsize=(8, 7), dpi=120, layout="constrained")
    ax = fig.add_subplot(projection="3d")
    source_xyz, target_xyz, aligned_xyz = map(coordinates, (source, target, aligned))
    ax.scatter(*source_xyz.T, color="#94a3b8", s=10, label="source")
    ax.scatter(*target_xyz.T, color="#0284c7", s=18, marker="x", label="target")
    ax.scatter(*aligned_xyz.T, color="#f43f5e", s=10, label="aligned source")
    for p, q in zip(aligned_xyz, target_xyz):
        ax.plot(*np.stack([p, q]).T, color="#f43f5e", linewidth=0.5, alpha=0.6)
    ax.set_title(title)
    ax.set_box_aspect([1, 1, 1])
    ax.legend(loc="upper left", fontsize=8)
    return fig
