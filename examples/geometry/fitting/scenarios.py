"""Scenes for the fitting example: one per figure, each returning the geometry to draw.

Every scene samples a primitive in a standard position, moves the samples by one shared
pose, and fits. The truth is the same pose applied to the standard primitive.
"""

from __future__ import annotations

import numpy as np

from examples.geometry.fitting.core import (
    Line, Plane, Point, bundle, cloud, fit, jitter, end_planes, mv, patch, patch_edges, segment,
)

pose = (mv.xw * 0.4 - mv.yw * 0.3 + mv.zw * 0.6).exp() * (mv.yz * 0.3).exp() * (mv.xy * 0.5).exp()


def point_to_points():
    """Fit the centroid of a noisy point cloud."""
    rng = np.random.default_rng(0)
    points = jitter(pose >> cloud(200, 0.5, rng), 0.05, rng)
    centroid = fit(Point, points)
    return points, pose >> mv.zyx, centroid


def line_to_points():
    """Fit the principal line through a noisy segment; both lines are clipped by two end planes."""
    rng = np.random.default_rng(0)
    points = jitter(pose >> segment(120, 2.0), 0.05, rng)
    line = fit(Line, points)
    ends = pose >> end_planes(2.5)
    return points, (pose >> mv.xz) ^ ends, line ^ ends


def plane_to_points():
    """Fit a plane through a noisy patch; both planes are cut into quads by four edge lines."""
    rng = np.random.default_rng(0)
    points = jitter(pose >> patch(200, 2.0, rng), 0.05, rng)
    plane = fit(Plane, points)
    edges = pose >> patch_edges(2.0)
    return points, (pose >> mv.z) ^ edges, plane ^ edges


def point_to_lines():
    """Triangulate the point of closest approach to a bundle of lines.

    Each line is drawn as a segment through the fitted point, from its end points.
    """
    rng = np.random.default_rng(0)
    rays = pose >> bundle(30, 0.05, rng)
    intersection = fit(Point, rays).normalized()

    # Each line's point at infinity is its direction. An ideal point has no weight to
    # normalize; its length is the norm of its dual, a Euclidean vector.
    directions = rays ^ mv.w
    unit_directions = directions / directions.dual().norm()
    ends = intersection + unit_directions * np.array([[-2.0], [2.0]])
    return ends, pose >> mv.zyx, intersection


if __name__ == "__main__":
    from examples.animation import save_figure
    from examples.geometry.fitting import render

    save_figure(render.draw_point_fit(*point_to_points()), "fitting_point_to_points")
    save_figure(render.draw_line_fit(*line_to_points()), "fitting_line_to_points")
    save_figure(render.draw_plane_fit(*plane_to_points()), "fitting_plane_to_points")
    save_figure(render.draw_bundle_fit(*point_to_lines()), "fitting_point_to_lines")
