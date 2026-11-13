"""Tests for rendering quadric ellipsoids on the 3-sphere by projection."""

from __future__ import annotations

import numpy as np

from examples.quadrics.s3_raytracer import render, scenarios
from examples.quadrics.s3_raytracer.core import Plane, direction, mv, origin, pixel_chart, project, reproject


def test_reprojected_hits_lie_on_the_ellipsoid():
    """Depths from the screen conic reconstruct points on the surface; misses are NaN."""
    axes = direction(np.eye(3))
    ellipsoid = (axes * (axes & Plane) * 0.04).sum(axis=0) - origin * (origin & Plane)
    placement = (mv.xw * 0.4).exp()
    surface = (placement >> ellipsoid(placement << Plane)).inverse()
    conic, polar = project(mv.rotor(), surface)
    chart = pixel_chart(np.radians(60.0), (30, 40))
    depth = reproject(conic, polar, chart)
    hit = ~depth.isnan()
    assert 0 < hit.sum() < hit.size
    points = (origin * depth[hit] + chart[hit]).normalized()
    np.testing.assert_allclose((points & surface(points)).to_array(), 0.0, atol=1e-8)


def test_one_frame_of_the_walk_draws():
    scene = scenarios.walk(1)
    render.draw_walk(*scene, (32, 48))
    next(render.frames(*scene, (32, 48)))
