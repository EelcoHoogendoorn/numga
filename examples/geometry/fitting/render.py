"""GPU rendering for projective primitive fitting draws the samples as blue balls or rods, the true primitive in grey, and the fitted primitive in red."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from numga import stack
from numga.render import solids
from numga.topology import Topology
from examples.geometry.fitting.core import Point, mv
from examples.util.panels import image_figure

sample_colour = np.array([0.42, 0.58, 0.74])
true_colour = np.array([0.55, 0.56, 0.60])
fit_colour = np.array([0.90, 0.28, 0.30])
background = np.array([1.0, 1.0, 1.0])
# The view's elevation and azimuth, the pixels across, and the margin around what it shows as a fraction of its
# larger side.
elevation, azimuth = np.radians(24.0), np.radians(32.0)
pixels = 520
margin = 0.08
# Radii of a sample, of a fitted or true point, and of a line or edge drawn.
sample_radius, point_radius, line_radius = 0.035, 0.09, 0.022
quad_topology = Topology.loops([4])


def euclidean_coordinates(points: Point) -> np.ndarray:
    """Euclidean coordinates (`[..., 3]`) read out each point's pairings with the coordinate planes divided by its weight."""
    return ((stack((mv.x, mv.y, mv.z)) & points[..., None]) / (mv.w & points)[..., None]).to_array()


def scene_figure(framed: list[Point], reach: float, parts: list[tuple[solids.Quadric, np.ndarray]]) -> plt.Figure:
    """Render a figure of coloured quadric batches framed around the given point sets."""
    view = solids.view(framed, reach, elevation, azimuth, margin, pixels)
    return image_figure(solids.draw(parts, view, background, inks=[], translucent=[], meshes=[]))


def sample_balls(centres: Point, radius: float) -> solids.Quadric:
    """Sphere quadrics (`[n_centres, 1]`) sit at `centres` with `radius`."""
    return solids.ball(centres.reshape(-1), radius)[:, None]


def point_fit_figure(points: Point, truth: Point, fitted_point: Point) -> plt.Figure:
    """Draw the noisy point cloud in blue with the true centre in grey and the fitted centroid in red."""
    return scene_figure(
        [points, truth, fitted_point],
        point_radius,
        [
            (sample_balls(points, sample_radius), sample_colour),
            (sample_balls(truth, point_radius), true_colour),
            (sample_balls(fitted_point, point_radius), fit_colour),
        ],
    )


def line_fit_figure(points: Point, truth_ends: Point, fit_ends: Point) -> plt.Figure:
    """Draw the noisy sample points in blue with the true line in grey and the fitted line in red."""
    return scene_figure(
        [points, truth_ends, fit_ends],
        line_radius,
        [
            (sample_balls(points, sample_radius), sample_colour),
            (solids.rods(truth_ends[:1], truth_ends[1:], line_radius), true_colour),
            (solids.rods(fit_ends[:1], fit_ends[1:], line_radius), fit_colour),
        ],
    )


def plane_fit_figure(points: Point, truth_quad: Point, fit_quad: Point) -> plt.Figure:
    """Draw the noisy sample points in blue with the true plane in grey and the fitted plane in red as quad outlines."""
    def quad_rods(corners: Point) -> solids.Quadric:
        tails, heads = (quad_topology.incidence[1, 0] * corners.field()).batch()        # [4] Point-, Point-
        return solids.rods(tails, heads, line_radius)

    return scene_figure(
        [points, truth_quad, fit_quad],
        line_radius,
        [
            (sample_balls(points, sample_radius), sample_colour),
            (quad_rods(truth_quad), true_colour),
            (quad_rods(fit_quad), fit_colour),
        ],
    )


def bundle_fit_figure(ends: Point, truth: Point, fitted_point: Point) -> plt.Figure:
    """Draw the line bundle in blue with the true point in grey and the fitted point in red."""
    return scene_figure(
        [ends, truth, fitted_point],
        point_radius,
        [
            (solids.rods(ends[0], ends[1], line_radius / 2), sample_colour),
            (sample_balls(truth, point_radius), true_colour),
            (sample_balls(fitted_point, point_radius), fit_colour),
        ],
    )
