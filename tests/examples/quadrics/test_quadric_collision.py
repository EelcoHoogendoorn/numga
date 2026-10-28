"""Tests for ellipsoid collision from blends of dual quadrics in PGA2D."""

from __future__ import annotations

import numpy as np

from examples.quadrics.quadric_collision import render, scenarios
from examples.quadrics.quadric_collision.core import Plane, cubic_peak, ellipsoid, motor, mv, blend, tangent_plane


def test_dual_ellipsoid_tangency():
    """Tangent planes from the support formula satisfy planes & Q(planes) == 0 for every normal."""
    placement = motor(1.5, -0.8, np.radians(35))
    Q = placement >> ellipsoid(2.0, 1.2)(placement << Plane)
    phi = np.radians([0, 45, 90, 135, 210, 315])
    planes = tangent_plane(Q, mv.x * np.cos(phi) + mv.y * np.sin(phi))
    np.testing.assert_allclose((planes & Q(planes)).to_array(), 0.0, atol=1e-10)


def test_tangent_plane_ignores_the_scale_of_the_quadric():
    """Q and any nonzero multiple of it, negative included, are the same quadric with the same tangents."""
    placement = motor(1.5, -0.8, np.radians(35))
    Q = placement >> ellipsoid(2.0, 1.2)(placement << Plane)
    phi = np.radians([0, 45, 90, 135, 210, 315])
    normals = mv.x * np.cos(phi) + mv.y * np.sin(phi)
    np.testing.assert_allclose(tangent_plane(Q * -2.5, normals).kernel, tangent_plane(Q, normals).kernel, atol=1e-12)


def test_cubic_peak_matches_a_dense_sweep():
    """The closed-form peak of the blend's determinant agrees with a sampled maximum."""
    placement = motor(2.5, 0.7, np.radians(-20))
    Q1 = ellipsoid(2.0, 1.0)
    Q2 = placement >> ellipsoid(1.0, 0.5)(placement << Plane)
    samples = mv.scalar(np.array([[0.0], [1.0], [2.0], [-1.0]]))
    parameter, maximum = cubic_peak(blend(Q1, Q2, samples).dual().det())
    sweep = np.linspace(0.0, 1.0, 20001)
    values = blend(Q1, Q2, mv.scalar(sweep[:, None])).dual().det().to_array()
    np.testing.assert_allclose(parameter.to_array(), sweep[values.argmax()], atol=0.001)
    np.testing.assert_allclose(maximum.to_array(), values.max(), atol=1e-8)


def test_the_three_poses_are_apart_touching_and_overlapping():
    """The blend's peak determinant is positive apart, zero touching and negative overlapping."""
    maximum = scenarios.collision()[3].to_array()
    assert maximum[0] > 0 > maximum[2]
    np.testing.assert_allclose(maximum[1], 0, atol=1e-12)


def test_scenario_renders():
    frames = list(render.animate_collision(*scenarios.slide(scenarios.SLIDE[::14])))
    assert len(frames) == len(scenarios.SLIDE[::14])
    assert np.any(frames[0] != frames[-1])
