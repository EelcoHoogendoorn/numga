"""Render the animated tracking drive with the true, dead-reckoned, and filtered paths, the pose readings, and the filter's two-standard-deviation position quadric."""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from numga import concatenate, stack
from examples.util.animation import capture
from examples.util.wgsl import pinned, planar, room
from examples.util.wgsl.planar import SOLID, Cap, Curves, Marker, Marks, Polygons, Regions, Strokes
from examples.estimation.kalman.core import Motor, Point, mv

TRUE_COLOUR, DEAD_COLOUR, FILTER_COLOUR, READING_COLOUR = "#1f2329", "#d0482b", "#2c6fb4", "#3a9a5b"
FILTER_RGB = np.array(mcolors.to_rgb(FILTER_COLOUR))
# A vehicle's outline in its own frame, pointing along +x; the view's margin around the drive as a fraction
# of its larger side, the figure's width in inches, and pixels per point at 100 dpi.
OUTLINE = [[0.32, 0.0], [-0.18, 0.16], [-0.18, -0.16]]
MARGIN = 0.12
WIDTH = 3.2
POINT = 100 / 72
# The trails' widths in pixels, the dead-reckoned trail's dashes, the readings' marks' reach and ring width, and the
# vehicles' opacity.
TRUE_WIDTH, DEAD_WIDTH, FILTER_WIDTH = 1.3 * POINT, 1.0 * POINT, 1.0 * POINT
DEAD_DASHES = (5.1, 2.2)
READING_REACH, READING_RING = 4.0, 1.5
VEHICLE_OPACITY = 0.9


# --- plumbing -------------------------------------------------------------------------
def rgb(colour: str) -> np.ndarray:
    """A named colour's red, green and blue."""
    return np.array(mcolors.to_rgb(colour))


def animate_tracking(truth: Motor, dead: Motor, estimates: Motor, positions: Point, ellipsoids,
                     readings: Motor) -> Iterator[np.ndarray]:
    """Yield one frame per state of the drive with the trails of the true, dead-reckoned, and filtered paths, the pose readings, the implicit position quadric, and the three vehicles."""
    origin = mv.w.dual()
    true_path, dead_path = truth >> origin, dead >> origin                     # [steps] Point-, [steps] Point-
    outline = mv.point(OUTLINE)
    outlines = {colour: poses[..., None] >> outline
                for colour, poses in ((TRUE_COLOUR, truth), (DEAD_COLOUR, dead), (FILTER_COLOUR, estimates))}
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
    # Lay the figure out once; each frame draws the trails so far, the readings, the quadric, and the vehicles.
    for step in range(len(true_path)):
        layers = [Regions(ellipsoids[step:step + 1], FILTER_RGB, 0.18),
                  Curves(ellipsoids[step:step + 1], 1.0 * POINT, FILTER_RGB, 1.0),
                  Strokes(true_path[None, :step + 1], TRUE_WIDTH, rgb(TRUE_COLOUR), 1.0, SOLID, Cap.ROUND),
                  Strokes(dead_path[None, :step + 1], DEAD_WIDTH, rgb(DEAD_COLOUR), 1.0, DEAD_DASHES, Cap.BUTT),
                  Strokes(positions[None, :step + 1], FILTER_WIDTH, FILTER_RGB, 1.0, SOLID, Cap.ROUND),
                  Marks(reading_points[:step + 1][fresh[:step + 1]], Marker.DIAMOND, READING_REACH - READING_RING,
                        READING_REACH, rgb(READING_COLOUR), 1.0),
                  *[Polygons(shape[step:step + 1], rgb(colour), VEHICLE_OPACITY) for colour, shape in outlines.items()]]
        image.set_data(planar.draw(view, layers, np.ones(3)))
        yield capture(fig)
    plt.close(fig)
