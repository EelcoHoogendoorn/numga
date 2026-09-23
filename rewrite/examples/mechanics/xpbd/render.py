"""Drawing for the XPBD chain: link positions at a few moments, and the joint gaps over time."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.mechanics.xpbd.core import Point, Scalar


def euclidean(points: Point) -> np.ndarray:
    """Read xyz coordinates using an explicit coordinate basis, independent of storage order."""
    k = np.asarray(points.cast(points.algebra.subspace("yzw zxw xyw zyx")).kernel)
    return k[..., :3] / k[..., 3:]


def draw_chain(centres: Point, gaps: Scalar) -> plt.Figure:
    """Render link positions at five moments, and the history of the joint gaps."""
    fig = plt.figure(figsize=(12, 5), dpi=120)

    # 1. 3D link trajectory
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    xyz = euclidean(centres)                                 # [steps, links, 3]
    last = len(xyz) - 1
    moments = [0, last // 4, last // 2, 3 * last // 4, last]
    colors = plt.cm.viridis(np.linspace(0.2, 1.0, len(moments)))
    for moment, color in zip(moments, colors):
        ax1.plot(*xyz[moment].T, "-o", color=color, label=f"Step {moment}", markersize=5)
    ax1.set_title("Swinging Chain (XPBD Rigid Body Poses)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")
    ax1.legend(loc="upper left", fontsize=8)

    # 2. Joint separation history
    ax2 = fig.add_subplot(1, 2, 2)
    separation = np.asarray(gaps.to_array())                 # [steps, joints]
    ax2.plot(separation.max(axis=1), label="Max joint separation", color="crimson", linewidth=1.5)
    ax2.plot(separation.mean(axis=1), label="Mean joint separation", color="royalblue", linestyle="--")
    ax2.set_yscale("log")
    ax2.set_title("Constraint Violation History", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Time step")
    ax2.set_ylabel("Anchor distance (m)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    return fig
