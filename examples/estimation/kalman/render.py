"""Render the animated tracking drive with the true, dead-reckoned, and filtered paths, the pose readings, and the filter's two-standard-deviation position quadric."""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from numga import concatenate, stack
from examples.util.animation import capture
from examples.util.wgsl import pinned, planar, room
from examples.util.wgsl.planar import Curves, Regions
from examples.estimation.kalman.core import Motor, Point, mv

TRUE_COLOUR, DEAD_COLOUR, FILTER_COLOUR, READING_COLOUR = "#1f2329", "#d0482b", "#2c6fb4", "#3a9a5b"
FILTER_RGB = np.array(mcolors.to_rgb(FILTER_COLOUR))
# A vehicle's outline in its own frame, pointing along +x; the view's margin around the drive as a fraction
# of its larger side, the figure's width in inches, and pixels per point at 100 dpi.
OUTLINE = [[0.32, 0.0], [-0.18, 0.16], [-0.18, -0.16]]
MARGIN = 0.12
WIDTH = 3.2
POINT = 100 / 72


# --- plumbing -------------------------------------------------------------------------
def coordinates(points: Point) -> np.ndarray:
    """Read out the x and y coordinates of each point from its pairings with the coordinate planes."""
    return (stack((mv.x, mv.y)) & points.normalized()[..., None]).to_array()


def animate_tracking(truth: Motor, dead: Motor, estimates: Motor, positions: Point, ellipsoids,
                     readings: Motor) -> Iterator[np.ndarray]:
    """Yield one frame per state of the drive with the trails of the true, dead-reckoned, and filtered paths, the pose readings, the implicit position quadric, and the three vehicles."""
    origin = mv.w.dual()
    true_xy, dead_xy, est_xy, reading_xy = coordinates(truth >> origin), coordinates(dead >> origin), coordinates(positions), coordinates(readings >> origin)
    outline = mv.point(OUTLINE)
    outlines = {colour: poses[..., None] >> outline
                for colour, poses in ((TRUE_COLOUR, truth), (DEAD_COLOUR, dead), (FILTER_COLOUR, estimates))}
    shapes = {colour: coordinates(shape) for colour, shape in outlines.items()}
    # A reading is new where its position differs from the one before.
    reading_points = readings >> origin
    fresh = np.r_[False, (reading_points[1:] - reading_points[:-1]).dual().norm() > 0]
    everything = concatenate([truth >> origin, dead >> origin, positions, reading_points[fresh],
                              *[shape.reshape(-1) for shape in outlines.values()]])
    heights = stack((mv.x, mv.y))[:, None] & everything.normalized()
    low = heights.min(axis=-1)
    high = heights.max(axis=-1)
    spans = high - low
    spans = spans + 2 * MARGIN * spans.max()
    aspect = float((spans[1] / spans[0]).to_array())
    fig = plt.figure(figsize=(WIDTH, WIDTH * aspect), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    view = planar.view([everything], MARGIN, room(ax))
    image = pinned(ax, np.full(view.shape + (3,), 255, dtype=np.uint8), view.extent)
    # Lay the figure out once; only the trails, the readings, the quadric image, and the vehicles move between frames.
    true_trail, = ax.plot([], [], color=TRUE_COLOUR, linewidth=1.3)
    dead_trail, = ax.plot([], [], color=DEAD_COLOUR, linestyle="--", linewidth=1.0)
    est_trail, = ax.plot([], [], color=FILTER_COLOUR, linewidth=1.0)
    reading_marks, = ax.plot([], [], marker="x", color=READING_COLOUR, linestyle="none", markersize=5, markeredgewidth=1.4)
    vehicles = {colour: ax.fill(*shape[0].T, color=colour, alpha=0.9, linewidth=0)[0]
                for colour, shape in shapes.items()}
    for step in range(len(true_xy)):
        layers = [Regions(ellipsoids[step:step + 1], FILTER_RGB, 0.18),
                  Curves(ellipsoids[step:step + 1], 1.0 * POINT, FILTER_RGB, 1.0)]
        image.set_data(planar.draw(view, layers, np.ones(3)))
        true_trail.set_data(*true_xy[:step + 1].T)
        dead_trail.set_data(*dead_xy[:step + 1].T)
        est_trail.set_data(*est_xy[:step + 1].T)
        reading_marks.set_data(*reading_xy[:step + 1][fresh[:step + 1]].T)
        for colour, shape in shapes.items():
            vehicles[colour].set_xy(shape[step])
        yield capture(fig)
    plt.close(fig)
