"""Animated rendering of the true, dead-reckoned, and Kalman-filtered vehicle paths alongside sparse pose readings and the two-standard-deviation position confidence quadric."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from examples.estimation.kalman.core import Covariance, Motor, Point, Quadric, mv, origin
from examples.util.animation import capture
from examples.util.panels import overlay, panel
from numga import concatenate, stack
from numga.render import Cap, contours, drawing, quadrics, simplices
from numga.topology import Topology

true_colour, dead_colour, filter_colour, reading_colour = "#1f2329", "#d0482b", "#2c6fb4", "#3a9a5b"
filter_rgb = np.array(mcolors.to_rgb(filter_colour))
# Vehicle triangle vertices in the body frame pointing along +x, view margin fraction, figure width in inches, and pixels per point at 100 dpi.
outline_coordinates = [[0.32, 0.0], [-0.18, 0.16], [-0.18, -0.16]]
margin = 0.12
width_inches = 3.2
point_px = 100 / 72
# Trail widths in pixels, dead-reckoned dash pattern, reading marker radius, and vehicle fill opacity.
true_width, dead_width, filter_width = 1.3 * point_px, 1.0 * point_px, 1.0 * point_px
dead_dashes = (5.1, 2.2)
reading_radius = 3.0
vehicle_opacity = 0.9


# --- plumbing -------------------------------------------------------------------------
def rgb_colour(colour: str) -> np.ndarray:
    """Hex colour `colour` converts to a `[3]` RGB float array."""
    return np.array(mcolors.to_rgb(colour))


def drive_window(increments) -> tuple[np.ndarray, float]:
    """Commanded twist `increments` (`[n_readings, n_steps_per_reading] Twist+`) set a nominal camera box (`[2, 2]`) and
    height-to-width aspect ratio around the meandering trajectory."""
    pose = mv.rotor()                                                           # [] Motor+
    nominal = [origin]
    for increment in increments.reshape(-1):
        pose = pose * (increment * 0.5).exp()                                   # [] Motor+
        nominal.append(pose >> origin)
    outline = mv.point(outline_coordinates)                                     # [3] Point-
    enclosing = concatenate([stack(nominal), (pose >> outline)])
    heights = stack((mv.x, mv.y))[:, None] & enclosing.normalized()             # [2, ...] Scalar+
    low = heights.min(axis=-1)                                                  # [2] Scalar+
    high = heights.max(axis=-1)                                                 # [2] Scalar+
    spans = high - low                                                          # [2] Scalar+
    # Extra breathing room around the nominal commanded path keeps noisy excursions and initial ellipses inside the frame.
    middle, half = (low + high) / 2, spans / 2 + 2.2 * margin * spans.max()     # [2] Scalar+, [2] Scalar+
    limits = stack((middle - half, middle + half), axis=-1).to_array()
    aspect = float((half[1] / half[0]).to_array())
    return limits, aspect


def tracking_animation(
    increments,
    ellipsoid_of: Callable[[Motor, Covariance, Point], Quadric],
    states: Iterable[tuple[Motor, Motor, Motor, Covariance, Motor]],
) -> Iterator[np.ndarray]:
    """Yields one RGB frame per state from `states`, accumulating the true, dead-reckoned, and filtered trails, the
    received pose readings, the implicit position confidence quadric, and the three vehicle outlines."""
    limits, aspect = drive_window(increments)
    figure = plt.figure(figsize=(width_inches, width_inches * aspect), dpi=100)
    ax = figure.add_axes([0, 0, 1, 1])
    ax.set(xlim=limits[0], ylim=limits[1])
    camera = panel(ax, mv.x, mv.y, mv.w)
    image = overlay(ax, [np.zeros(camera.shape + (4,))])
    outline = mv.point(outline_coordinates)                                     # [3] Point-
    vehicle = Topology(3, [[0, 1, 2]])

    true_points: list[Point] = []
    dead_points: list[Point] = []
    filtered_points: list[Point] = []
    reading_points: list[Point] = []
    last_reading_point: Point | None = None

    for truth, dead, estimate, sigma, reading in states:
        true_points.append(truth >> origin)
        dead_points.append(dead >> origin)
        position = estimate >> origin                                           # [] Point-
        filtered_points.append(position)
        reading_point = reading >> origin                                       # [] Point-
        if last_reading_point is not None and bool((reading_point - last_reading_point).dual().norm() > 0):
            reading_points.append(reading_point)
        last_reading_point = reading_point

        ellipsoid = ellipsoid_of(estimate, sigma, position)[None]               # [1] Plane+ <- Point-
        trail = Topology.chains([len(true_points)])
        true_trail = stack(true_points)
        dead_trail = stack(dead_points)
        filtered_trail = stack(filtered_points)
        pictures = [
            quadrics.regions(ellipsoid[:, None], filter_rgb, camera, 0.18),
            quadrics.curves(ellipsoid, 1.0 * point_px, filter_rgb, camera),
            simplices.strokes(true_trail, trail, rgb_colour(true_colour), true_width, camera),
            simplices.strokes(dead_trail, trail, rgb_colour(dead_colour), dead_width, camera,
                              dashes=dead_dashes, cap=Cap.BUTT),
            simplices.strokes(filtered_trail, trail, filter_rgb, filter_width, camera),
        ]
        if reading_points:
            pictures.append(simplices.markers(stack(reading_points), rgb_colour(reading_colour), reading_radius, camera))
        for colour, pose in ((true_colour, truth), (dead_colour, dead), (filter_colour, estimate)):
            pictures.append(
                contours.picture(pose >> outline, vehicle, np.zeros(3), rgb_colour(colour), camera, opacity=vehicle_opacity)
            )
        image.set_data(drawing.composited(pictures))
        yield capture(figure)
    plt.close(figure)

