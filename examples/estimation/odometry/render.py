"""Render the true, dead-reckoned, and most likely trajectories around the lap with each pose's confidence ellipsoid on the GPU."""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from numga import concatenate, stack
from numga.algebras import PGA2D
from examples import instantiate
from examples.util.animation import capture
from examples.util.wgsl import quadrics, simplices
from examples.util.wgsl.camera import Camera, panel
from examples.util.wgsl.simplices import Cap, Topology

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
def extent(truth: core.Motor, dead: core.Motor) -> np.ndarray:
    """The lower and upper limits across and up of a box around the true and dead-reckoned positions, with a margin,
    `[2, 2]`."""
    positions = concatenate([truth, dead]) >> origin
    readings = stack((mv.x, mv.y))[:, None] & positions.normalized()
    low = readings.min(axis=-1)
    high = readings.max(axis=-1)
    spans = high - low
    margin = 0.15 * spans.max()
    return stack((low - margin, high + margin), axis=-1).to_array()


def path(points: Point, width: float, colour: np.ndarray, camera: Camera, dashes: tuple[float, float] = (1.0, 0.0),
         cap: Cap = Cap.PROJECTING) -> np.ndarray:
    """A polyline through points `[points]`, `width` points across."""
    return simplices.strokes(points, Topology.chains([len(points)]), colour, width * POINT, camera, dashes=dashes, cap=cap)


def disc(points: Point, width: float, colour: np.ndarray, camera: Camera) -> np.ndarray:
    """Discs at points `[points]`, `width` points across with an outline a point wide."""
    return simplices.markers(points, colour, (width + 1.0) / 2 * POINT, camera)


def base_pictures(truth: core.Motor, dead: core.Motor, reckoned_ellipsoids, camera: Camera) -> list[np.ndarray]:
    """The true path, the dashed dead-reckoned path, and its uncertainty ellipsoids."""
    dash_pattern = (DASHED[0] * DEAD_WIDTH * POINT, DASHED[1] * DEAD_WIDTH * POINT)
    return [
        path(truth >> origin, TRUTH_WIDTH, TRUTH_RGB, camera),
        path(dead >> origin, DEAD_WIDTH, DEAD_RGB, camera, dash_pattern, Cap.BUTT),
        quadrics.curves(reckoned_ellipsoids, ELLIPSOID_WIDTH * POINT, DEAD_RGB, camera, 0.35),
    ]


def likely_pictures(truth: core.Motor, poses: core.Motor, ellipsoids, camera: Camera) -> list[np.ndarray]:
    """The estimated poses, their uncertainty ellipsoids, and the start marker."""
    positions = poses >> origin                                                 # [poses] Point-
    return [
        quadrics.curves(ellipsoids, ELLIPSOID_WIDTH * POINT, LIKELY_RGB, camera, 0.7),
        path(positions, LIKELY_WIDTH, LIKELY_RGB, camera),
        disc(positions, LIKELY_MARK, LIKELY_RGB, camera),
        disc(truth[:1] >> origin, START_MARK, START_RGB, camera),
    ]


def framed(limits: np.ndarray) -> tuple[plt.Figure, plt.Axes, Camera]:
    """Create a figure, axes on the box between the limits `[2, 2]`, and the camera its picture is read by."""
    figure, ax = plt.subplots(figsize=(5, 5))
    ax.set_axis_off()
    figure.tight_layout()
    ax.set(xlim=limits[0], ylim=limits[1])
    return figure, ax, panel(ax, mv.x, mv.y, mv.w)


def draw_lap(truth: core.Motor, dead: core.Motor, reckoned_ellipsoids, linked_poses: core.Motor) -> plt.Figure:
    """Render the true lap, the dead-reckoned trajectory with its uncertainty ellipsoids, and the loop-closing link."""
    figure, ax, camera = framed(extent(truth, dead))
    linked_positions = linked_poses >> origin                                   # [endpoints] Point-
    simplices.overlay(ax, base_pictures(truth, dead, reckoned_ellipsoids, camera) + [
        path(linked_positions, CLOSING_WIDTH, CLOSING_RGB, camera),
        disc(linked_positions, CLOSING_MARK, CLOSING_RGB, camera),
        disc(truth[:1] >> origin, START_MARK, START_RGB, camera),
    ])
    return figure


def draw(truth: core.Motor, dead: core.Motor, reckoned_ellipsoids, poses: core.Motor,
         ellipsoids) -> plt.Figure:
    """Render the true path, the dead-reckoned trajectory with its uncertainty ellipsoids, and the estimated poses with theirs."""
    figure, ax, camera = framed(extent(truth, dead))
    simplices.overlay(ax, base_pictures(truth, dead, reckoned_ellipsoids, camera)
                      + likely_pictures(truth, poses, ellipsoids, camera))
    return figure


def animate(truth: core.Motor, dead: core.Motor, reckoned_ellipsoids, closing_steps) -> Iterator[np.ndarray]:
    """Yield one frame per damped Gauss-Newton iteration over the true path and dead reckoning."""
    figure, ax, camera = framed(extent(truth, dead))
    background = base_pictures(truth, dead, reckoned_ellipsoids, camera)
    image = simplices.overlay(ax, background)
    for poses, ellipsoids in closing_steps:
        image.set_data(simplices.composited(background + likely_pictures(truth, poses, ellipsoids, camera)))
        yield capture(figure)
    plt.close(figure)
