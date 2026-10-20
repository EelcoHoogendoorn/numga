"""Drawing for the robot arm: link boxes, the target, and the trail of the tip."""

from __future__ import annotations

from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np

from examples.animation import capture
from numga import stack
from examples.mechanics.robot_arm.core import Point, mv

BOX_EDGES = [(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


# --- plumbing -------------------------------------------------------------------------
def euclidean(points: Point) -> np.ndarray:
    """Coordinates of points: their pairings with the planes x, y and z, over their weights."""
    return ((stack((mv.x, mv.y, mv.z)) & points[..., None]) / (mv.w & points)[..., None]).to_array()


def draw_arm(ax, boxes: Point, target: Point, trail: list[np.ndarray]) -> None:
    """Draw the link boxes, the target, and the tip's trail so far."""
    ax.cla()
    for xyz, color in zip(euclidean(boxes), ("tab:blue", "tab:cyan", "tab:purple")):
        for a, b in BOX_EDGES:
            ax.plot(*zip(xyz[a], xyz[b]), color=color, linewidth=1.2)
    ax.scatter(*euclidean(target), color="tab:red", s=40)
    ax.plot(*np.array(trail).T, color="tab:red", linewidth=0.8, alpha=0.6)
    ax.set_xlim(-0.5, 2.0); ax.set_ylim(-1.25, 1.25); ax.set_zlim(0, 3.0); ax.set_box_aspect((1, 1, 1.2))
    ax.view_init(elev=22, azim=-50)


def animate_tracking(states: Iterable) -> list[np.ndarray]:
    """One frame per solved state: the link boxes, the target, and the tip trail."""
    fig = plt.figure(figsize=(5, 5), dpi=100)
    ax = fig.add_subplot(projection="3d")
    frames, trail = [], []
    for boxes, target, tip in states:
        trail.append(euclidean(tip))
        draw_arm(ax, boxes, target, trail)
        frames.append(capture(fig))
    plt.close(fig)
    return frames
