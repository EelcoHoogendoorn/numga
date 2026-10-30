"""A rotating crank, its applied force and handle velocity beside a mirror."""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from numga import stack
from examples.util.animation import capture
from examples.orientation.mechanical_work.core import Point

velocity_colour = "#218c9d"
force_colour = "#ce713f"
crank_colour = "#424c56"
guide_colour = "#d4dce1"
mirror_colour = "#aebfca"
# The figure's width and height in pixels, at the dots per inch its line widths and marker sizes are given at.
figure_pixels = (420, 210)
dpi = 100
# The widths of the mirror, the handles' paths, the cranks and the pivots' rims, in points; the areas of the pivots
# and the handles, in square points.
mirror_width, guide_width, crank_width, rim_width = 0.8, 0.6, 2.6, 0.95
pivot_area, handle_area = 33, 15


# --- plumbing -------------------------------------------------------------------------
def coordinates(points: Point) -> np.ndarray:
    """x and y of points `[..., 2]`: their pairings with the planes x and y."""
    mv = points.context.multivector
    return (stack((mv.x, mv.y)) & points.normalized()[..., None]).to_array()


def components(directions: Point) -> np.ndarray:
    """x and y of directions `[..., 2]`: their pairings with the planes x and y."""
    mv = directions.context.multivector
    return (stack((mv.x, mv.y)) & directions[..., None]).to_array()


class WorkView:
    """The mirror, the handles' paths and the pivots, set up once; the cranks, their handles and the arrows at them,
    set for each state."""

    def __init__(self, pivots: Point, paths: Point) -> None:
        self.pivots = pivots.normalized()                                       # [cranks] Point-
        self.centres = coordinates(pivots)
        trails = coordinates(paths)
        # The longest crank: the distance from a pivot to its handle is the norm of their join.
        radii = (self.pivots & paths.normalized()).norm().reshape(-1)
        margin = radii.max() * 0.95
        mv = paths.context.multivector
        readings = stack((mv.x, mv.y))[:, None] & paths.normalized().reshape(-1)
        lower = (readings.min(axis=-1) - margin).to_array()
        upper = (readings.max(axis=-1) + margin).to_array()

        self.figure, ax = plt.subplots(figsize=np.array(figure_pixels) / dpi, dpi=dpi, layout="constrained")
        ax.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]), aspect="equal")
        ax.set_axis_off()
        ax.axvline(0, color=mirror_colour, linewidth=mirror_width, linestyle=(0, (4, 5)), zorder=0)
        closed_trails = np.concatenate([trails, trails[:1]], axis=0)
        ax.add_collection(LineCollection(closed_trails.transpose(1, 0, 2), colors=guide_colour, linewidths=guide_width))
        self.cranks = LineCollection(np.stack([self.centres, self.centres], axis=1), colors=crank_colour,
                                     linewidths=crank_width)
        ax.add_collection(self.cranks)
        ax.scatter(*self.centres.T, s=pivot_area, color=crank_colour, edgecolors="white", linewidths=rim_width,
                   zorder=3)
        self.handles = ax.scatter(*self.centres.T, s=handle_area, color=crank_colour, zorder=3)
        # The velocity's arrows and the force's, each on one side of the handles.
        self.arrows = [(ax.quiver(*self.centres.T, *np.zeros_like(self.centres).T, angles="xy", scale_units="xy",
                                  scale=1, color=colour, width=0.007, headwidth=3.8, headlength=4.8, zorder=4), side)
                       for colour, side in ((velocity_colour, -1), (force_colour, 1))]

    def update(self, handles: Point, velocities: Point, forces: Point) -> plt.Figure:
        handles = handles.normalized()                                          # [cranks] Point-
        radii = (self.pivots & handles).norm()
        radius = radii.max()
        # The direction out from each pivot to its handle, per length of the longest crank.
        radial = (handles - self.pivots) / radius                               # [cranks] Direction-
        # The fastest handle's velocity reaches over half a crank: a direction's size is the norm of its join with a
        # unit point.
        speeds = (handles & velocities).norm()
        arrow_scale = radius * 0.55 / speeds.max()
        offset = radius * 0.055
        points = coordinates(handles)
        self.cranks.set_segments(np.stack([self.centres, points], axis=1))
        self.handles.set_offsets(points)
        # Separate the glyphs slightly so equal directions remain visible.
        for (arrows, side), directions in zip(self.arrows, (velocities, forces)):
            arrows.set_offsets(coordinates(handles + side * offset * radial))
            arrows.set_UVC(*components(directions * arrow_scale).T)
        return self.figure


def draw_work(pivots: Point, handles: Point, velocities: Point, forces: Point, paths: Point) -> plt.Figure:
    return WorkView(pivots, paths).update(handles, velocities, forces)


def animate(pivots: Point, handles: Point, velocities: Point, forces: Point) -> Iterator[np.ndarray]:
    """The handles turn through the supplied geometric states."""
    view = WorkView(pivots, handles)
    for handle, velocity, force in zip(handles, velocities, forces):
        yield capture(view.update(handle, velocity, force))
    plt.close(view.figure)
