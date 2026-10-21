"""Drawing for the fitting example, ray traced: the samples as small spheres, with the true primitive, grey,
and the fitted one, red."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from numga import stack
from examples import raytrace as rt
from examples.geometry.fitting.core import Point, mv

SAMPLE_COLOUR, TRUE_COLOUR, FIT_COLOUR = np.array([0.42, 0.58, 0.74]), np.array([0.55, 0.56, 0.60]), np.array([0.90, 0.28, 0.30])
BACKGROUND = np.array([1.0, 1.0, 1.0])
# The view's angles, the pixels across, the margin around what it shows as a fraction of its larger side,
# and the figure's width in inches.
VIEW = (24, -58)
PIXELS = 520
MARGIN = 0.08
WIDTH = 6.0
# Radii of a sample, of a fitted or true point, and of a line or edge drawn.
SAMPLE_RADIUS, POINT_RADIUS, LINE_RADIUS = 0.035, 0.09, 0.022


# --- plumbing -------------------------------------------------------------------------
def euclidean(points: Point) -> np.ndarray:
    """Coordinates of points: their pairings with the planes x, y and z, over their weights."""
    return ((stack((mv.x, mv.y, mv.z)) & points[..., None]) / (mv.w & points)[..., None]).to_array()


def ball(centres: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
    """Balls of a radius about centres `[n, 3]`: their matrices and centres."""
    return np.broadcast_to(np.eye(3) / radius**2, centres.shape + (3,)), centres


def draw(samples: np.ndarray, parts: list[tuple[tuple[np.ndarray, np.ndarray], np.ndarray]]) -> plt.Figure:
    """One figure: the samples `[n, 3]` as small spheres, and the parts, each solid quadrics given by their
    matrices and centres with one colour, framed by the samples' and the parts' surfaces."""
    matrices = np.concatenate([matrix for (matrix, _), _ in parts])
    centres = np.concatenate([centre for (_, centre), _ in parts])
    surfaces = np.concatenate([rt.ellipsoid_points(*ball(samples, SAMPLE_RADIUS), 12).reshape(-1, 3),
                               rt.ellipsoid_points(matrices, centres, 48).reshape(-1, 3)])
    middle, half = rt.framing(surfaces, *VIEW, MARGIN)
    view = (*VIEW, middle, half, PIXELS * rt.SUPERSAMPLE)
    rgb, depth = rt.spheres(samples, SAMPLE_RADIUS, np.tile(SAMPLE_COLOUR, (len(samples), 1)), view)
    rgb = np.where(np.isfinite(depth)[..., None], rgb, BACKGROUND)
    colours = np.concatenate([np.tile(colour, (len(centre), 1)) for (_, centre), colour in parts])
    solid, solid_depth = rt.scene(rt.quadric(matrices, centres), colours, view, BACKGROUND)
    image = rt.downsample(np.where((solid_depth < depth)[..., None], solid, rgb), rt.SUPERSAMPLE)
    fig = plt.figure(figsize=(WIDTH, WIDTH * image.shape[0] / image.shape[1]), dpi=PIXELS / WIDTH)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(np.clip(image, 0, 1), interpolation="bilinear")
    ax.set_axis_off()
    return fig


def draw_point_fit(points: Point, truth: Point, fit: Point) -> plt.Figure:
    """The noisy cloud, blue, with the true centre, grey, and the fitted one, red."""
    samples, true_centre, fit_centre = euclidean(points).reshape(-1, 3), euclidean(truth)[None], euclidean(fit)[None]
    return draw(samples, [(ball(true_centre, POINT_RADIUS), TRUE_COLOUR), (ball(fit_centre, POINT_RADIUS), FIT_COLOUR)])


def draw_line_fit(points: Point, truth_ends: Point, fit_ends: Point) -> plt.Figure:
    """The noisy points, blue, with the true line, grey, and the fitted one, red, each drawn between its two
    end points."""
    true_ends, found_ends = euclidean(truth_ends), euclidean(fit_ends)
    return draw(euclidean(points).reshape(-1, 3),
                [(rt.segments(true_ends[:1], true_ends[1:], LINE_RADIUS), TRUE_COLOUR),
                 (rt.segments(found_ends[:1], found_ends[1:], LINE_RADIUS), FIT_COLOUR)])


def draw_plane_fit(points: Point, truth_quad: Point, fit_quad: Point) -> plt.Figure:
    """The noisy points, blue, with the true plane, grey, and the fitted one, red, each drawn as the edges of
    a quad."""
    true_corners, found_corners = euclidean(truth_quad), euclidean(fit_quad)
    return draw(euclidean(points).reshape(-1, 3),
                [(rt.segments(true_corners, np.roll(true_corners, -1, axis=0), LINE_RADIUS), TRUE_COLOUR),
                 (rt.segments(found_corners, np.roll(found_corners, -1, axis=0), LINE_RADIUS), FIT_COLOUR)])


def draw_bundle_fit(ends: Point, truth: Point, fit: Point) -> plt.Figure:
    """The bundle, blue, each line drawn between its paired end points, with the true point, grey, and the
    fitted one, red."""
    starts, finishes = euclidean(ends)
    true_point, found_point = euclidean(truth)[None], euclidean(fit)[None]
    return draw(np.zeros((0, 3)),
                [(rt.segments(starts, finishes, LINE_RADIUS / 2), SAMPLE_COLOUR),
                 (ball(true_point, POINT_RADIUS), TRUE_COLOUR), (ball(found_point, POINT_RADIUS), FIT_COLOUR)])
