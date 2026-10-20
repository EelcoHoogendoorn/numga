"""Drawing for the XPBD chain: the links swinging, and the joint gaps over time."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from numga import stack
from examples.animation import capture
from examples.mechanics.xpbd.core import Point, Scalar

# The view's margin around the chain's whole swing, and its angles.
MARGIN = 1.1
ELEVATION, AZIMUTH = 20, -60


# --- plumbing -------------------------------------------------------------------------
def euclidean(points: Point) -> np.ndarray:
    """Coordinates of points: their pairings with the planes x, y and z, over their weights."""
    mv = points.context.multivector
    return np.asarray(((stack((mv.x, mv.y, mv.z)) & points[..., None]) / (mv.w & points)[..., None]).to_array())


def animate_chain(centres: Point) -> list[np.ndarray]:
    """One frame per step: the links' centres joined in order, the fixed link at the origin."""
    xyz = euclidean(centres)                                                   # [steps + 1, links, 3]
    # A cube about the whole swing.
    low, high = xyz.reshape(-1, 3).min(axis=0), xyz.reshape(-1, 3).max(axis=0)
    centre, half_width = (low + high) / 2, MARGIN * (high - low).max() / 2
    fig = plt.figure(figsize=(5, 5), dpi=100)
    ax = fig.add_subplot(projection="3d")
    frames = []
    for links in xyz:
        ax.cla()
        ax.plot(*links.T, "-o", color="#2c5d9e", markersize=5)
        ax.set(xlim=centre[0] + np.array([-half_width, half_width]), ylim=centre[1] + np.array([-half_width, half_width]),
               zlim=centre[2] + np.array([-half_width, half_width]))
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=ELEVATION, azim=AZIMUTH)
        frames.append(capture(fig))
    plt.close(fig)
    return frames


def draw_gaps(gaps: Scalar) -> plt.Figure:
    """The largest and the mean joint gap at each step, on a log scale."""
    separation = np.asarray(gaps.to_array())                                   # [steps + 1, joints]
    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
    ax.plot(separation.max(axis=1), color="crimson", linewidth=1.5)
    ax.plot(separation.mean(axis=1), color="royalblue", linestyle="--")
    ax.set_yscale("log")
    ax.set(xlabel="time step", ylabel="anchor distance (m)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
