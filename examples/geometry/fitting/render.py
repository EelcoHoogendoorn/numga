"""Drawing for the fitting example, ray traced on the GPU: the samples as small balls, with the true primitive,
grey, and the fitted one, red."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from numga import stack
from examples.geometry.fitting.core import Point, mv
from examples.wgsl import image_figure, solids

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
    """Coordinates of points: their pairings with the planes x, y and z, over their weights."""
    return ((stack((mv.x, mv.y, mv.z)) & points[..., None]) / (mv.w & points)[..., None]).to_array()


def draw(framed: list[Point], reach: float, parts: list[tuple[solids.Quadric, np.ndarray]]) -> plt.Figure:
    """One figure of solids in parts, each a batch of solids with one colour, framed around batches of points
    with `reach` to spare."""
    return image_figure(solids.draw(parts, solids.view(framed, reach, elevation, azimuth, margin, pixels), background))


def balls(centres: Point, radius: float) -> solids.Quadric:
    return solids.ball(centres.reshape(-1), radius)[:, None]


def draw_point_fit(points: Point, truth: Point, fit: Point) -> plt.Figure:
    """The noisy cloud, blue, with the true centre, grey, and the fitted one, red."""
    return draw([points, truth, fit], point_radius,
                [(balls(points, sample_radius), sample_colour),
                 (balls(truth, point_radius), true_colour), (balls(fit, point_radius), fit_colour)])


def draw_line_fit(points: Point, truth_ends: Point, fit_ends: Point) -> plt.Figure:
    """The noisy points, blue, with the true line, grey, and the fitted one, red, each drawn between its two
    end points."""
    return draw([points, truth_ends, fit_ends], line_radius,
                [(balls(points, sample_radius), sample_colour),
                 (solids.rods(truth_ends[:1], truth_ends[1:], line_radius), true_colour),
                 (solids.rods(fit_ends[:1], fit_ends[1:], line_radius), fit_colour)])


def draw_plane_fit(points: Point, truth_quad: Point, fit_quad: Point) -> plt.Figure:
    """The noisy points, blue, with the true plane, grey, and the fitted one, red, each drawn as the edges of
    a quad."""
    def edges(corners: Point) -> solids.Quadric:
        return solids.rods(corners, corners[np.roll(np.arange(len(corners)), -1)], line_radius)
    return draw([points, truth_quad, fit_quad], line_radius,
                [(balls(points, sample_radius), sample_colour),
                 (edges(truth_quad), true_colour), (edges(fit_quad), fit_colour)])


def draw_bundle_fit(ends: Point, truth: Point, fit: Point) -> plt.Figure:
    """The bundle, blue, each line drawn between its paired end points, with the true point, grey, and the
    fitted one, red."""
    return draw([ends, truth, fit], point_radius,
                [(solids.rods(ends[0], ends[1], line_radius / 2), sample_colour),
                 (balls(truth, point_radius), true_colour), (balls(fit, point_radius), fit_colour)])
