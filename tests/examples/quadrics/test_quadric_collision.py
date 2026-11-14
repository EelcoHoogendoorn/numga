"""Tests for ellipsoid collision from blends of dual quadrics in PGA2D."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.quadrics.quadric_collision import render, scenarios
from examples.quadrics.quadric_collision.core import (
    Plane, cubic_peak, mv, placement_motor, quadric_blend, rest_ellipsoid, tangent_plane,
)


def test_dual_ellipsoid_tangency_and_scale_invariance():
    """Outward tangent planes satisfy `planes & dual(planes) == 0` and ignore nonzero rescaling of `dual`."""
    placement = placement_motor(1.5, -0.8, np.radians(35))
    shape = rest_ellipsoid(2.0, 1.2)
    dual = placement >> shape(placement << Plane)
    angles = np.radians([0, 45, 90, 135, 210, 315])
    rotors = (mv.xy * (-angles / 2)).exp()
    normals = rotors >> mv.x
    planes = tangent_plane(dual, normals)
    assert ((planes & dual(planes)).abs() < 1e-10).all()
    assert ((tangent_plane(dual * -2.5, normals) - planes).norm() < 1e-12).all()


def test_cubic_peak_matches_a_sampled_sweep():
    """The closed-form cubic peak of the blend determinant agrees with a dense sweep on `(0, 1)`."""
    placement = placement_motor(2.5, 0.7, np.radians(-20))
    first = rest_ellipsoid(2.0, 1.0)
    second_shape = rest_ellipsoid(1.0, 0.5)
    second = placement >> second_shape(placement << Plane)
    samples = mv.scalar([[0], [1], [2], [-1]])
    peak_share, peak_det = cubic_peak(quadric_blend(first, second, samples).dual().det())
    sweep = np.linspace(0.0, 1.0, 2001)
    values = quadric_blend(first, second, mv.scalar(sweep[:, None])).dual().det()
    best = values.argmax()
    assert (peak_share - sweep[best]).abs() < 1e-3
    assert (peak_det - values[best]).abs() < 1e-6


def test_three_poses_and_contact():
    """The peak determinant is positive apart, zero touching, and negative overlapping, with a shared tangent and contact point."""
    first_ellipsoid, placed_second, _, peak_dets, contact_plane, contact_point = scenarios.posed_pair()
    assert peak_dets[0] > 0 and peak_dets[2] < 0
    assert peak_dets[1].abs() < 1e-12
    assert (contact_plane & first_ellipsoid(contact_plane)).abs() < 1e-12
    assert (contact_plane & placed_second[1](contact_plane)).abs() < 1e-12
    assert (placed_second[1](contact_plane).normalized() & contact_point).norm() < 1e-12


def test_one_frame_of_collision_animation_draws():
    """Streaming a single slide state through `render.collision_animation` yields an RGB frame."""
    frames = render.collision_animation(*scenarios.slide_setup(scenarios.slide_offsets[:1]))
    frame = next(frames)
    assert frame.ndim == 3 and frame.shape[2] == 3
    frames.close()
    plt.close("all")
