"""Tests for rendering quadric ellipsoids on the 3-sphere by projection."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.quadrics.s3_raytracer import render, scenarios
from examples.quadrics.s3_raytracer.core import (
    mv,
    origin,
    outlines,
    pixel_chart,
    placed_ellipsoids,
    project,
    reproject,
    walk_frames,
)


def test_reprojected_hits_lie_on_the_ellipsoid():
    """Cotangent depths from the screen conic reconstruct surface hits while misses stay `NaN`."""
    surface = placed_ellipsoids(
        np.array([0.2, 0.2, 0.2]),
        np.array([0.8]),
        np.array([0.0]),
        np.array([0.0]),
        0.0,
    )[0]
    cone = outlines(mv.rotor(), surface)
    conic, polar = project(mv.rotor(), surface)
    chart = pixel_chart(np.radians(60.0), (24, 32))
    depths = reproject(conic, polar, chart)
    hit_mask = ~depths.isnan()
    assert 0 < hit_mask.sum() < hit_mask.size
    assert ((chart[hit_mask] & cone(chart[hit_mask])) >= -1e-6).all()
    points = (origin * depths[hit_mask] + chart[hit_mask]).normalized()
    assert ((points & surface(points)).abs() < 1e-6).all()


def test_one_frame_of_the_walk_draws():
    """Rendering one static view and one streamed walk frame succeeds at small resolution."""
    surfaces, body_colours, light, fov = scenarios.scene_setup()
    frames = walk_frames(scenarios.walk_distance, 1)
    first_frame = next(frames)
    fig = render.walk_figure(first_frame, surfaces, body_colours, light, fov, (24, 32))
    plt.close(fig)
    image = next(render.walk_animation(surfaces, body_colours, light, fov, (24, 32), [first_frame]))
    assert image.shape == (24, 32, 3)
