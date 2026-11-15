"""GPU rendering of the true, dead-reckoned, and estimated pose trajectories around the lap with implicit confidence ellipsoids."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from examples import instantiate
from examples.util.animation import capture
from examples.util.panels import overlay, panel
from numga import concatenate, stack
from numga.algebras import PGA2D
from numga.render import Camera, Cap, drawing, quadrics, simplices
from numga.topology import Topology

# Drawings render the planar lap in PGA2D.
core = instantiate("examples.estimation.odometry.core", PGA2D)
mv = core.mv
Point = PGA2D.gatype.antivector()                   # Antivector-
origin = mv.w.dual()                                # [] Point-
# Palette for dead reckoning, the estimated poses, the true path, the loop-closing link, and the known start.
dead_colour, likely_colour = "#d9822b", "#2c5d9e"
truth_colour, closing_colour, start_colour = "0.75", "#2a9d4a", "0.2"
dead_rgb, likely_rgb, truth_rgb, closing_rgb, start_rgb = (
    np.array(mcolors.to_rgb(colour))
    for colour in (dead_colour, likely_colour, truth_colour, closing_colour, start_colour)
)
# Pixels per typographic point at 100 dpi, stroke widths in points, dash-gap multiples, and marker widths in points.
point_px = 100 / 72
truth_width, dead_width, likely_width, closing_width, ellipsoid_width = 3.0, 1.4, 1.0, 1.5, 0.8
dashed_pattern = (3.7, 1.6)
likely_mark, closing_mark, start_mark = 2.5, 5.0, 7.0


# --- plumbing -------------------------------------------------------------------------
def frame_limits(truth: core.Motor, dead: core.Motor) -> np.ndarray:
    """Bounding coordinates (`[2, 2]`) enclose the true and dead-reckoned positions with a uniform margin."""
    positions = concatenate([truth, dead]) >> origin                            # [...] Point-
    readings = stack((mv.x, mv.y))[:, None] & positions.normalized()            # [2, ...] Scalar+
    low = readings.min(axis=-1)                                                 # [2] Scalar+
    high = readings.max(axis=-1)                                                # [2] Scalar+
    spans = high - low                                                          # [2] Scalar+
    margin = 0.15 * spans.max()                                                 # [] Scalar+
    return stack((low - margin, high + margin), axis=-1).to_array()


def path_layer(
    points: Point,
    width: float,
    colour: np.ndarray,
    camera: Camera,
    dashes: tuple[float, float] = (1.0, 0.0),
    cap: Cap = Cap.PROJECTING,
) -> np.ndarray:
    """Draws an open polyline through `points` (`[...] Point-`) at `width` points across."""
    return simplices.strokes(
        points, Topology.chains([len(points)]), colour, width * point_px, camera, dashes=dashes, cap=cap,
    )


def marker_layer(points: Point, width: float, colour: np.ndarray, camera: Camera) -> np.ndarray:
    """Draws circular markers at `points` (`[...] Point-`) sized `width` points across plus a 1-point outline."""
    return simplices.markers(points, colour, (width + 1.0) / 2 * point_px, camera)


def base_layers(truth: core.Motor, dead: core.Motor, reckoned_ellipsoids: core.Quadric, camera: Camera) -> list:
    """Forms the background layers showing the true path, dashed dead reckoning, and dead-reckoned ellipsoids."""
    dashes = (dashed_pattern[0] * dead_width * point_px, dashed_pattern[1] * dead_width * point_px)
    return [
        path_layer(truth >> origin, truth_width, truth_rgb, camera),
        path_layer(dead >> origin, dead_width, dead_rgb, camera, dashes, Cap.BUTT),
        quadrics.curves(reckoned_ellipsoids, ellipsoid_width * point_px, dead_rgb, camera, 0.35),
    ]


def estimate_layers(truth: core.Motor, poses: core.Motor, ellipsoids: core.Quadric, camera: Camera) -> list:
    """Forms the foreground layers showing the estimated poses, their confidence ellipsoids, and the start marker."""
    positions = poses >> origin                                                 # [...] Point-
    return [
        quadrics.curves(ellipsoids, ellipsoid_width * point_px, likely_rgb, camera, 0.7),
        path_layer(positions, likely_width, likely_rgb, camera),
        marker_layer(positions, likely_mark, likely_rgb, camera),
        marker_layer(truth[:1] >> origin, start_mark, start_rgb, camera),
    ]


def camera_panel(limits: np.ndarray) -> tuple[plt.Figure, plt.Axes, Camera]:
    """Creates a borderless `5` by `5` inch figure, axes over `limits` (`[2, 2]`), and its `PGA2D` camera."""
    figure, ax = plt.subplots(figsize=(5, 5))
    ax.set_axis_off()
    figure.tight_layout()
    ax.set(xlim=limits[0], ylim=limits[1])
    return figure, ax, panel(ax, mv.x, mv.y, mv.w)


def dead_reckoning_figure(
    truth: core.Motor,
    dead: core.Motor,
    reckoned_ellipsoids: core.Quadric,
    linked_poses: core.Motor,
) -> plt.Figure:
    """Renders the true lap, the dead-reckoned trajectory with its confidence ellipsoids, and the loop-closing link."""
    figure, ax, camera = camera_panel(frame_limits(truth, dead))
    linked_positions = linked_poses >> origin                                   # [2] Point-
    overlay(ax, base_layers(truth, dead, reckoned_ellipsoids, camera) + [
        path_layer(linked_positions, closing_width, closing_rgb, camera),
        marker_layer(linked_positions, closing_mark, closing_rgb, camera),
        marker_layer(truth[:1] >> origin, start_mark, start_rgb, camera),
    ])
    return figure


def trajectory_figure(
    truth: core.Motor,
    dead: core.Motor,
    reckoned_ellipsoids: core.Quadric,
    poses: core.Motor,
    ellipsoids: core.Quadric,
) -> plt.Figure:
    """Renders the true path, dead reckoning with its confidence ellipsoids, and the estimated poses with theirs."""
    figure, ax, camera = camera_panel(frame_limits(truth, dead))
    overlay(
        ax,
        base_layers(truth, dead, reckoned_ellipsoids, camera) + estimate_layers(truth, poses, ellipsoids, camera),
    )
    return figure


def closing_animation(
    truth: core.Motor,
    dead: core.Motor,
    reckoned_ellipsoids: core.Quadric,
    ellipsoids_of: Callable[[core.Motor, core.Covariance], core.Quadric],
    states: Iterable[tuple[core.Motor, core.Covariance]],
) -> Iterator[np.ndarray]:
    """Yields one RGB frame per `(poses, uncertainty)` state in `states` over the true path and dead reckoning."""
    figure, ax, camera = camera_panel(frame_limits(truth, dead))
    background = base_layers(truth, dead, reckoned_ellipsoids, camera)
    image = overlay(ax, background)
    for poses, uncertainty in states:
        layers = background + estimate_layers(truth, poses, ellipsoids_of(poses, uncertainty), camera)
        image.set_data(drawing.composited(layers))
        yield capture(figure)
    plt.close(figure)
