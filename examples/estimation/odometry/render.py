"""Render the true, dead-reckoned, and most likely trajectories around the lap with each pose's confidence ellipsoid on the GPU via `planar`."""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from numga import concatenate, stack
from numga.algebras import PGA2D
from examples import instantiate
from examples.util.animation import capture
from examples.util.wgsl import pinned, planar, room
from examples.util.wgsl.planar import SOLID, Cap, Curves, Marker, Marks, Strokes, Window

# The drawings are of the lap in PGA2D.
core = instantiate("examples.estimation.odometry.core", PGA2D)
mv = core.mv
Point = PGA2D.gatype.antivector()                   # Antivector-
# The point each pose carries: the origin, dual to the weight direction w.
origin = mv.w.dual()                                # [] Point-
# The colours of dead reckoning, the most likely poses, the true path, the closing reading, and the known start.
DEAD, LIKELY = "#d9822b", "#2c5d9e"
TRUTH, CLOSING, START = "0.75", "#2a9d4a", "0.2"
DEAD_RGB, LIKELY_RGB, TRUTH_RGB, CLOSING_RGB, START_RGB = (
    np.array(mcolors.to_rgb(colour)) for colour in (DEAD, LIKELY, TRUTH, CLOSING, START)
)
# Pixels a point, at the figure's 100 dots per inch. The widths of the true, dead-reckoned and most likely paths, the
# closing reading and the ellipsoids; the dead-reckoned path's dashes, on and off, in widths of its stroke; and the
# widths of the marks of the most likely poses, the closing reading and the start, their outlines a point wide.
POINT = 100 / 72
TRUTH_WIDTH, DEAD_WIDTH, LIKELY_WIDTH, CLOSING_WIDTH, ELLIPSOID_WIDTH = 3.0, 1.4, 1.0, 1.5, 0.8
DASHED = (3.7, 1.6)
LIKELY_MARK, CLOSING_MARK, START_MARK = 2.5, 5.0, 7.0


# --- plumbing -------------------------------------------------------------------------
def coordinates(points: Point) -> np.ndarray:
    """Read out the x and y coordinates of each point from its pairings with the coordinate planes."""
    return (stack((mv.x, mv.y)) & points.normalized()[..., None]).to_array()


def extent(truth: core.Motor, dead: core.Motor) -> Point:
    """The lower and upper corner points of a box around the true and dead-reckoned positions, with a margin."""
    positions = concatenate([truth, dead]) >> origin
    axes = stack((mv.x, mv.y))
    readings = axes[:, None] & positions.normalized()
    low = readings.min(axis=-1)
    high = readings.max(axis=-1)
    spans = high - low
    margin = 0.15 * spans.max()
    return origin + (axes.dual() * stack((low - margin, high + margin))).sum(axis=-1)


def base_layers(truth: core.Motor, dead: core.Motor, reckoned_ellipsoids) -> list:
    """Build the planar layers for the true path, the dashed dead-reckoned path, and its uncertainty ellipsoids."""
    dash_pattern = (DASHED[0] * DEAD_WIDTH * POINT, DASHED[1] * DEAD_WIDTH * POINT)
    return [
        Strokes((truth >> origin)[None], TRUTH_WIDTH * POINT, TRUTH_RGB, 1.0, SOLID, Cap.PROJECTING),
        Strokes((dead >> origin)[None], DEAD_WIDTH * POINT, DEAD_RGB, 1.0, dash_pattern, Cap.BUTT),
        Curves(reckoned_ellipsoids, ELLIPSOID_WIDTH * POINT, DEAD_RGB, 0.35),
    ]


def likely_layers(truth: core.Motor, poses: core.Motor, ellipsoids) -> list:
    """Build the planar layers for the estimated poses, their uncertainty ellipsoids, and the start marker."""
    positions = poses >> origin                                                 # [poses] Point-
    return [
        Curves(ellipsoids, ELLIPSOID_WIDTH * POINT, LIKELY_RGB, 0.7),
        Strokes(positions[None], LIKELY_WIDTH * POINT, LIKELY_RGB, 1.0, SOLID, Cap.PROJECTING),
        Marks(positions, Marker.DISC, 0.0, (LIKELY_MARK + 1.0) / 2 * POINT, LIKELY_RGB, 1.0),
        Marks(truth[:1] >> origin, Marker.DISC, 0.0, (START_MARK + 1.0) / 2 * POINT, START_RGB, 1.0),
    ]


def framed(box: Point) -> tuple[plt.Figure, plt.Axes, Window]:
    """Create a figure, axes, and planar window for the bounding box."""
    figure, ax = plt.subplots(figsize=(5, 5))
    ax.set_axis_off()
    figure.tight_layout()
    view = planar.view([box], 0.0, room(ax))
    return figure, ax, view


def draw_lap(truth: core.Motor, dead: core.Motor, reckoned_ellipsoids, linked_poses: core.Motor) -> plt.Figure:
    """Render the true lap, the dead-reckoned trajectory with its uncertainty ellipsoids, and the loop-closing link."""
    figure, ax, view = framed(extent(truth, dead))
    linked_positions = linked_poses >> origin                                   # [endpoints] Point-
    layers = base_layers(truth, dead, reckoned_ellipsoids) + [
        Strokes(linked_positions[None], CLOSING_WIDTH * POINT, CLOSING_RGB, 1.0, SOLID, Cap.PROJECTING),
        Marks(linked_positions, Marker.DISC, 0.0, (CLOSING_MARK + 1.0) / 2 * POINT, CLOSING_RGB, 1.0),
        Marks(truth[:1] >> origin, Marker.DISC, 0.0, (START_MARK + 1.0) / 2 * POINT, START_RGB, 1.0),
    ]
    pinned(ax, planar.draw(view, layers, np.ones(3)), view.extent)
    return figure


def draw(truth: core.Motor, dead: core.Motor, reckoned_ellipsoids, poses: core.Motor,
         ellipsoids) -> plt.Figure:
    """Render the true path, the dead-reckoned trajectory with its uncertainty ellipsoids, and the estimated poses with theirs."""
    figure, ax, view = framed(extent(truth, dead))
    layers = base_layers(truth, dead, reckoned_ellipsoids) + likely_layers(truth, poses, ellipsoids)
    pinned(ax, planar.draw(view, layers, np.ones(3)), view.extent)
    return figure


def animate(truth: core.Motor, dead: core.Motor, reckoned_ellipsoids, closing_steps) -> Iterator[np.ndarray]:
    """Yield one frame per damped Gauss-Newton iteration over the true path and dead reckoning."""
    figure, ax, view = framed(extent(truth, dead))
    image = pinned(ax, np.full(view.shape + (3,), 255, dtype=np.uint8), view.extent)
    background_layers = base_layers(truth, dead, reckoned_ellipsoids)
    for poses, ellipsoids in closing_steps:
        image.set_data(planar.draw(view, background_layers + likely_layers(truth, poses, ellipsoids), np.ones(3)))
        yield capture(figure)
    plt.close(figure)
