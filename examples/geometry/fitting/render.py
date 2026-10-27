"""GPU rendering for projective primitive fitting draws the samples as blue balls or rods, the true primitive in grey, and the fitted primitive in red."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from numga import stack
from examples.geometry.fitting.core import Point, mv
from examples.util.wgsl import image_figure, solids

sample_colour, true_colour, fit_colour = np.array([0.42, 0.58, 0.74]), np.array([0.55, 0.56, 0.60]), np.array([0.90, 0.28, 0.30])
background = np.array([1.0, 1.0, 1.0])
# The view's elevation and azimuth, the pixels across, and the margin around what it shows as a fraction of its
# larger side.
elevation, azimuth = np.radians(24.0), np.radians(32.0)
pixels = 520
margin = 0.08
# Radii of a sample, of a fitted or true point, and of a line or edge drawn.
sample_radius, point_radius, line_radius = 0.035, 0.09, 0.022


# --- plumbing -------------------------------------------------------------------------
def euclidean(points: Point) -> np.ndarray:
    """The Euclidean coordinates of points are their pairings with the coordinate planes divided by their weights."""
    return ((stack((mv.x, mv.y, mv.z)) & points[..., None]) / (mv.w & points)[..., None]).to_array()


def draw(framed: list[Point], reach: float, parts: list[tuple[solids.Quadric, np.ndarray]]) -> plt.Figure:
    """Render a figure of coloured quadric batches framed around the given point sets."""
    return image_figure(solids.draw(parts, solids.view(framed, reach, elevation, azimuth, margin, pixels), background,
                                    inks=[], translucent=[], meshes=[]))


def balls(centres: Point, radius: float) -> solids.Quadric:
    return solids.ball(centres.reshape(-1), radius)[:, None]


def draw_point_fit(points: Point, truth: Point, fit: Point) -> plt.Figure:
    """Draw the noisy point cloud in blue with the true centre in grey and the fitted centroid in red."""
    return draw([points, truth, fit], point_radius,
                [(balls(points, sample_radius), sample_colour),
                 (balls(truth, point_radius), true_colour), (balls(fit, point_radius), fit_colour)])


def draw_line_fit(points: Point, truth_ends: Point, fit_ends: Point) -> plt.Figure:
    """Draw the noisy sample points in blue with the true line in grey and the fitted line in red."""
    return draw([points, truth_ends, fit_ends], line_radius,
                [(balls(points, sample_radius), sample_colour),
                 (solids.rods(truth_ends[:1], truth_ends[1:], line_radius), true_colour),
                 (solids.rods(fit_ends[:1], fit_ends[1:], line_radius), fit_colour)])


def draw_plane_fit(points: Point, truth_quad: Point, fit_quad: Point) -> plt.Figure:
    """Draw the noisy sample points in blue with the true plane in grey and the fitted plane in red as quad outlines."""
    def edges(corners: Point) -> solids.Quadric:
        return solids.rods(corners, corners[np.roll(np.arange(len(corners)), -1)], line_radius)
    return draw([points, truth_quad, fit_quad], line_radius,
                [(balls(points, sample_radius), sample_colour),
                 (edges(truth_quad), true_colour), (edges(fit_quad), fit_colour)])


def draw_bundle_fit(ends: Point, truth: Point, fit: Point) -> plt.Figure:
    """Draw the line bundle in blue with the true point in grey and the fitted point in red."""
    return draw([ends, truth, fit], point_radius,
                [(solids.rods(ends[0], ends[1], line_radius / 2), sample_colour),
                 (balls(truth, point_radius), true_colour), (balls(fit, point_radius), fit_colour)])
