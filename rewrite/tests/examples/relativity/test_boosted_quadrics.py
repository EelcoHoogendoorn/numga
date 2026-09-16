"""Unit tests for Boosted Celestial Quadrics and Relativistic Aberration."""

from __future__ import annotations

import numpy as np

from examples.relativity.boosted_quadrics import (
    mv,
    Vector,
    QuadricType,
    make_spherical_quadric_extensor,
    make_canonical_contour,
    project_to_screen,
    build_celestial_scene,
)


def test_quadric_extensor_construction():
    """Verify that canonical quadric extensors satisfy k . Q(k) == 0 in the rest frame."""
    th_x = np.radians(20.0)
    th_y = np.radians(12.0)
    Q = make_spherical_quadric_extensor(th_x, th_y)
    k = make_canonical_contour(th_x, th_y, n_pts=100)

    assert Q.gatype == QuadricType
    assert Q.kernel.shape == (4, 4)

    # Invariance on null cone: k . Q(k) == 0
    residuals = (k | Q(k)).kernel
    assert np.max(np.abs(residuals)) < 1e-14


def test_spatial_rotation_outermorphism():
    """Verify that spatial rotors preserve the quadric equation k_rot . Q_rot(k_rot) == 0."""
    th_x = np.radians(16.0)
    th_y = np.radians(8.0)
    Q = make_spherical_quadric_extensor(th_x, th_y)
    k = make_canonical_contour(th_x, th_y, n_pts=100)

    # Rotate by 35 degrees around yz
    R = (mv.yz * (np.radians(35.0) / 2.0)).exp()
    Q_rot = R >> Q(R << Vector)
    k_rot = R >> k

    residuals = (k_rot | Q_rot(k_rot)).kernel
    assert np.max(np.abs(residuals)) < 1e-14


def test_lorentz_boost_outermorphism():
    """Verify that Lorentz boosts strictly preserve k' . Q'(k') == 0 across a range of rapidities."""
    th_x = np.radians(22.0)
    th_y = np.radians(10.0)
    Q = make_spherical_quadric_extensor(th_x, th_y)
    k = make_canonical_contour(th_x, th_y, n_pts=120)

    # Also apply an initial spatial tilt
    R = (mv.zx * (np.radians(15.0) / 2.0)).exp()
    Q_world = R >> Q(R << Vector)
    k_world = R >> k

    # Test boosts with both positive and negative rapidities
    for zeta in [-1.2, -0.6, -0.2, 0.0, 0.3, 0.8, 1.1]:
        L = (mv.zt * (zeta / 2.0)).exp()
        Q_boosted = L >> Q_world(L << Vector)
        k_boosted = L >> k_world

        residuals = (k_boosted | Q_boosted(k_boosted)).kernel
        assert np.max(np.abs(residuals)) < 1e-13


def test_penrose_terrell_circle_scaling():
    """Verify that a central circle remains an exact circle under boosts with exact radius."""
    th_c = np.radians(12.0)
    k_circle = make_canonical_contour(th_c, th_c, n_pts=200)

    u0, v0 = project_to_screen(k_circle)
    r0 = np.max(np.sqrt(u0**2 + v0**2))
    expected_r0 = np.tan(th_c)
    assert np.isclose(r0, expected_r0, atol=1e-12)

    # Check exact relativistic aberration radius across multiple rapidities
    for zeta in [-0.5, -0.25, 0.0, 0.15, 0.45]:
        L = (mv.zt * (zeta / 2.0)).exp()
        k_boosted = L >> k_circle
        u, v = project_to_screen(k_boosted)

        # Exact formula for forward boost: z' = sinh(zeta) + cosh(zeta)*cos(th), x' = sin(th)
        expected_r = np.sin(th_c) / (np.sinh(zeta) + np.cosh(zeta) * np.cos(th_c))
        radii = np.sqrt(u**2 + v**2)

        # 1. Exact radius match
        assert np.allclose(radii, expected_r, atol=1e-12)
        # 2. Exact circularity: max radius == min radius to machine precision
        assert np.ptp(radii) < 1e-14


def test_build_celestial_scene():
    """Verify that the celestial scene builds valid quadrics and contours."""
    scene = build_celestial_scene()
    assert len(scene) >= 20

    for name, Q, k, color, lw in scene:
        assert isinstance(name, str)
        assert Q.gatype == QuadricType
        assert k.kernel.shape[-1] == 4
        # All rest-frame contours must satisfy k . Q(k) == 0
        res = (k | Q(k)).kernel
        assert np.max(np.abs(res)) < 1e-13


def test_perspective_vs_stereographic_circle_projection():
    """Verify why off-axis circles become perspective ellipses on flat sensors but stay circular stereographically."""
    # Off-axis circular cap tilted at 18 degrees
    th_c = np.radians(6.5)
    k_local = make_canonical_contour(th_c, th_c, n_pts=200)

    theta, phi = np.radians(18.0), np.radians(45.0)
    R = ((mv.zx * np.cos(phi) - mv.yz * np.sin(phi)) * (theta / 2.0)).exp()
    k_world = R >> k_local

    # 1. Perspective projection: oblique slice through circular cone produces ~5.2% stretch
    u_persp, v_persp = project_to_screen(k_world, projection_mode="perspective")
    major_span = np.ptp(u_persp + v_persp) / np.sqrt(2.0)
    minor_span = np.ptp(u_persp - v_persp) / np.sqrt(2.0)
    perspective_aspect_ratio = major_span / minor_span
    assert perspective_aspect_ratio > 1.04  # Demonstrates perspective elongation

    # 2. Stereographic projection: conformal map strictly preserves circles anywhere on S²
    # Test across multiple relativistic rapidities to verify Penrose-Terrell conformal preservation
    for zeta in [-0.8, 0.0, 0.6, 1.4]:
        L = (mv.zt * (zeta / 2.0)).exp()
        k_b = L >> k_world
        u_s, v_s = project_to_screen(k_b, projection_mode="stereographic")

        # Fit circle (u - u0)^2 + (v - v0)^2 = r^2
        A = np.stack([u_s, v_s, np.ones_like(u_s)], axis=-1)
        b = u_s**2 + v_s**2
        c, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        u0, v0 = c[0] / 2.0, c[1] / 2.0
        r = np.sqrt(c[2] + u0**2 + v0**2)
        dev = np.abs(np.sqrt((u_s - u0)**2 + (v_s - v0)**2) - r)
        rel_err = np.max(dev) / r
        assert rel_err < 1e-12  # Strict machine-precision circularity under Lorentz boost
