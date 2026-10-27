"""Scenes for projective primitive fitting sample a point cloud, a line segment, a planar patch, and a line bundle in a shared pose and fit the corresponding primitive to each."""

from __future__ import annotations

import numpy as np

from numga import stack
from examples.geometry.fitting.core import (
    Line, Plane, Point, bundle, cloud, fit, jitter, end_planes, mv, patch, patch_edges, segment,
)

# The pose every scene is placed by: a shift, after turns in the yz and xy planes.
SHIFT = np.array([0.4, -0.3, 0.6])
TILT, SPIN = 0.3, 0.5
POSE = ((stack((mv.xw, mv.yw, mv.zw)) * SHIFT).sum(axis=0)).exp() * (mv.yz * TILT).exp() * (mv.xy * SPIN).exp()
# The seed of every scene's samples, and the size of the noise that moves each sample point.
SEED = 0
NOISE = 0.05
# A cloud of points about the pose's origin.
N_CLOUD, CLOUD_SPREAD = 200, 0.5
# Points along a segment of the pose's y axis, and the half length the lines are drawn to.
N_SEGMENT, HALF_LENGTH, DRAWN_HALF_LENGTH = 120, 2.0, 2.5
# Points on a square of the pose's z == 0 plane.
N_PATCH, HALF_WIDTH = 200, 2.0
# Lines through points near the pose's origin, and the half length they are drawn to.
N_RAYS, RAY_SPREAD, RAY_HALF_LENGTH = 30, 0.05, 2.0


# --- math -----------------------------------------------------------------------------
def point_to_points():
    """A point fitted to a noisy cloud recovers its centroid."""
    rng = np.random.default_rng(SEED)
    points = jitter(POSE >> cloud(N_CLOUD, CLOUD_SPREAD, rng), NOISE, rng)     # [samples] Point-
    centroid = fit(Point, points)                                               # [] Point-
    return points, POSE >> mv.w.dual(), centroid


def line_to_points():
    """A line fitted to a noisy segment recovers its principal axis, clipped with the true line by two end planes."""
    rng = np.random.default_rng(SEED)
    points = jitter(POSE >> segment(N_SEGMENT, HALF_LENGTH), NOISE, rng)        # [samples] Point-
    line = fit(Line, points)                                                    # [] Line-
    truth = POSE >> (mv.w.dual() & mv.direction([0, 1, 0]))                     # [] Line-
    ends = POSE >> end_planes(DRAWN_HALF_LENGTH)                                # [2] Plane+
    return points, truth ^ ends, line ^ ends


def plane_to_points():
    """A plane fitted to a noisy patch recovers the patch plane, cut with the true plane into a quad by four edge lines."""
    rng = np.random.default_rng(SEED)
    points = jitter(POSE >> patch(N_PATCH, HALF_WIDTH, rng), NOISE, rng)        # [samples] Point-
    plane = fit(Plane, points)                                                  # [] Plane+
    edges = POSE >> patch_edges(HALF_WIDTH)                                     # [4] Line-
    return points, (POSE >> mv.z) ^ edges, plane ^ edges


def point_to_lines():
    """A point fitted to a bundle of lines triangulates their point of closest approach."""
    rng = np.random.default_rng(SEED)
    rays = POSE >> bundle(N_RAYS, RAY_SPREAD, rng)                              # [rays] Line-
    intersection = fit(Point, rays).normalized()                                # [] Point-

    # Each line's point at infinity is its direction. An ideal point has no weight to
    # normalize; its length is the norm of its dual, a Euclidean vector.
    directions = rays ^ mv.w                                                    # [rays] Point-
    unit_directions = directions / directions.dual().norm()
    ends = intersection + unit_directions * np.array([[-RAY_HALF_LENGTH], [RAY_HALF_LENGTH]])   # [2, rays] Point-
    return ends, POSE >> mv.w.dual(), intersection


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_figure
    from examples.geometry.fitting import render

    cloud_fit, line_fit, plane_fit, bundle_fit = point_to_points(), line_to_points(), plane_to_points(), point_to_lines()
    save_figure(render.draw_point_fit(*cloud_fit), "fitting_point_to_points")
    save_figure(render.draw_line_fit(*line_fit), "fitting_line_to_points")
    save_figure(render.draw_plane_fit(*plane_fit), "fitting_plane_to_points")
    save_figure(render.draw_bundle_fit(*bundle_fit), "fitting_point_to_lines")

    # checks
    # The samples limit each fit: the centre comes back within 10 cm, about the cloud's spread over the root
    # of its count; the fitted line's and plane's drawn ends lie within 5 cm of the truth, and so does the
    # triangulated point.
    _, truth, centroid = cloud_fit
    assert (centroid.normalized() & truth.normalized()).norm() < 0.1
    _, _, fit_ends = line_fit
    assert (fit_ends.normalized() & (POSE >> (mv.w.dual() & mv.direction([0, 1, 0]))).normalized()).norm().to_array().max() < 0.05
    _, _, fit_quad = plane_fit
    assert (fit_quad.normalized() & (POSE >> mv.z).normalized()).norm().to_array().max() < 0.05
    _, truth, intersection = bundle_fit
    assert (intersection.normalized() & truth.normalized()).norm() < 0.05


if __name__ == "__main__":
    main()
