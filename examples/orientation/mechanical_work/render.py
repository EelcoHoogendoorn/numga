"""A rotating crank, its applied force and handle velocity beside a mirror."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from examples.util.animation import capture
from examples.orientation.mechanical_work.core import Point, ga

VELOCITY = "#218c9d"
FORCE = "#ce713f"
FRAME = "#424c56"
GUIDE = "#d4dce1"
MIRROR = "#aebfca"


# --- plumbing -------------------------------------------------------------------------
def draw_work(pivots: Point, handles: Point, velocities: Point, forces: Point, paths: Point) -> plt.Figure:
    pivot_values = pivots.cast(ga.subspace("yw wx xy")).kernel
    centres = pivot_values[..., :2] / pivot_values[..., 2:]
    point_values = handles.cast(ga.subspace("yw wx xy")).kernel
    points = point_values[..., :2] / point_values[..., 2:]
    trail_values = paths.cast(ga.subspace("yw wx xy")).kernel
    trails = trail_values[..., :2] / trail_values[..., 2:]
    velocity = velocities.cast(ga.subspace("yw wx xy")).kernel[..., :2]
    force = forces.cast(ga.subspace("yw wx xy")).kernel[..., :2]

    radius = np.linalg.norm(points - centres, axis=-1).max()
    radial = (points - centres) / radius
    arrow_scale = radius * 0.55 / np.linalg.norm(velocity, axis=-1).max()
    offset = radius * 0.055
    margin = radius * 0.95
    lower = trails.min(axis=(0, 1)) - margin
    upper = trails.max(axis=(0, 1)) + margin

    figure, ax = plt.subplots(figsize=(8, 3.8), layout="constrained")
    ax.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]), aspect="equal")
    ax.set_axis_off()
    ax.axvline(0, color=MIRROR, linewidth=1.5, linestyle=(0, (4, 5)), zorder=0)
    closed_trails = np.concatenate([trails, trails[:1]], axis=0)
    ax.add_collection(LineCollection(closed_trails.transpose(1, 0, 2), colors=GUIDE, linewidths=1.1))
    ax.add_collection(LineCollection(np.stack([centres, points], axis=1), colors=FRAME, linewidths=5))
    ax.scatter(*centres.T, s=120, color=FRAME, edgecolors="white", linewidths=1.8, zorder=3)
    ax.scatter(*points.T, s=55, color=FRAME, zorder=3)

    # Separate the glyphs slightly so equal directions remain visible.
    for vectors, colour, side in ((velocity, VELOCITY, -1), (force, FORCE, 1)):
        starts = points + side * offset * radial
        ax.quiver(*starts.T, *(vectors * arrow_scale).T, angles="xy", scale_units="xy", scale=1,
                  color=colour, width=0.007, headwidth=3.8, headlength=4.8, zorder=4)
    return figure


def animate(pivots: Point, handles: Point, velocities: Point, forces: Point) -> list[np.ndarray]:
    """The handles turn through the supplied geometric states."""
    frames = []
    for handle, velocity, force in zip(handles, velocities, forces):
        figure = draw_work(pivots, handle, velocity, force, handles)
        frames.append(capture(figure))
        plt.close(figure)
    return frames
