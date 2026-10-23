"""A rotating crank, its applied force and handle velocity beside a mirror."""

from __future__ import annotations

from collections.abc import Iterator

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
def planar(points: Point) -> np.ndarray:
    """x and y of points `[..., 2]`."""
    values = points.cast(ga.subspace("yw wx xy")).kernel
    return values[..., :2] / values[..., 2:]


class WorkView:
    """The mirror, the handles' paths and the pivots, set up once; the cranks, their handles and the arrows at them,
    set for each state."""

    def __init__(self, pivots: Point, paths: Point) -> None:
        self.centres = planar(pivots)
        trails = planar(paths)
        radius = np.linalg.norm(trails - self.centres, axis=-1).max()
        margin = radius * 0.95
        lower = trails.min(axis=(0, 1)) - margin
        upper = trails.max(axis=(0, 1)) + margin

        self.figure, ax = plt.subplots(figsize=(8, 3.8), layout="constrained")
        ax.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]), aspect="equal")
        ax.set_axis_off()
        ax.axvline(0, color=MIRROR, linewidth=1.5, linestyle=(0, (4, 5)), zorder=0)
        closed_trails = np.concatenate([trails, trails[:1]], axis=0)
        ax.add_collection(LineCollection(closed_trails.transpose(1, 0, 2), colors=GUIDE, linewidths=1.1))
        self.cranks = LineCollection(np.stack([self.centres, self.centres], axis=1), colors=FRAME, linewidths=5)
        ax.add_collection(self.cranks)
        ax.scatter(*self.centres.T, s=120, color=FRAME, edgecolors="white", linewidths=1.8, zorder=3)
        self.handles = ax.scatter(*self.centres.T, s=55, color=FRAME, zorder=3)
        # The velocity's arrows and the force's, each on one side of the handles.
        self.arrows = [(ax.quiver(*self.centres.T, *np.zeros_like(self.centres).T, angles="xy", scale_units="xy",
                                  scale=1, color=colour, width=0.007, headwidth=3.8, headlength=4.8, zorder=4), side)
                       for colour, side in ((VELOCITY, -1), (FORCE, 1))]

    def update(self, handles: Point, velocities: Point, forces: Point) -> plt.Figure:
        points = planar(handles)
        velocity = velocities.cast(ga.subspace("yw wx xy")).kernel[..., :2]
        force = forces.cast(ga.subspace("yw wx xy")).kernel[..., :2]
        radius = np.linalg.norm(points - self.centres, axis=-1).max()
        radial = (points - self.centres) / radius
        arrow_scale = radius * 0.55 / np.linalg.norm(velocity, axis=-1).max()
        offset = radius * 0.055
        self.cranks.set_segments(np.stack([self.centres, points], axis=1))
        self.handles.set_offsets(points)
        # Separate the glyphs slightly so equal directions remain visible.
        for (arrows, side), vectors in zip(self.arrows, (velocity, force)):
            arrows.set_offsets(points + side * offset * radial)
            arrows.set_UVC(*(vectors * arrow_scale).T)
        return self.figure


def draw_work(pivots: Point, handles: Point, velocities: Point, forces: Point, paths: Point) -> plt.Figure:
    return WorkView(pivots, paths).update(handles, velocities, forces)


def animate(pivots: Point, handles: Point, velocities: Point, forces: Point) -> Iterator[np.ndarray]:
    """The handles turn through the supplied geometric states."""
    view = WorkView(pivots, handles)
    for handle, velocity, force in zip(handles, velocities, forces):
        yield capture(view.update(handle, velocity, force))
    plt.close(view.figure)
