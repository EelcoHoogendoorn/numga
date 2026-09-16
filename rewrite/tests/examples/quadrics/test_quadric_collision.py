"""Unit tests for 2D Dual Quadric (Ellipse) Collision Detection in PGA2D."""

import numpy as np
import pytest

from numga.algebras import PGA2D
from examples.quadrics.quadric_collision import (
    context,
    line_at_infinity,
    make_quadric,
    motor,
    tangent_line,
    find_contact_parameter,
    extract_contact_line_and_point,
    normalize_point,
)

mv = context.multivector
Lines = PGA2D.subspace.vector()


def test_dual_ellipse_tangency():
    """Verify that tangent lines evaluated via support formula satisfy L v Q(L) == 0."""
    Q_body = make_quadric(rx=2.0, ry=1.2)
    m = motor(tx=1.5, ty=-0.8, angle_rad=np.radians(35))
    Q = m >> Q_body(m << Lines)

    for angle_deg in [0, 45, 90, 135, 210, 315]:
        phi = np.radians(angle_deg)
        n_dir = (mv.x * np.cos(phi) + mv.y * np.sin(phi)).normalized()
        L = tangent_line(Q, n_dir)
        tangency = L.regressive(Q(L)).kernel.item()
        assert np.isclose(tangency, 0.0, atol=1e-10)


def test_collision_detection_three_states():
    """Verify the binary classification of separated, touching, and overlapping states."""

    Q1_body = make_quadric(rx=2.0, ry=1.0)
    m1 = motor(tx=-1.2, ty=0.0, angle_rad=np.radians(25))
    Q1 = m1 >> Q1_body(m1 << Lines)

    # Contact normal direction: 22 degrees
    phi = np.radians(22)
    n_dir = (mv.x * np.cos(phi) + mv.y * np.sin(phi)).normalized()
    L_target = tangent_line(Q1, n_dir)
    p_target = normalize_point(Q1(L_target))

    # Rotated Q2 body
    Q2_body = make_quadric(rx=1.6, ry=0.9)
    angle2 = np.radians(-35)
    m2_rot = motor(0.0, 0.0, angle2)
    Q2_rot = m2_rot >> Q2_body(m2_rot << Lines)
    L2_rot = tangent_line(Q2_rot, -n_dir)
    p2_rot = normalize_point(Q2_rot(L2_rot))

    # Tangential contact motor: pure PGA translation taking p2_rot to p_target
    disp = p_target - p2_rot
    T_touch = (line_at_infinity.wedge(disp.dual()) * -0.5).exp()
    m2_touch = T_touch * m2_rot
    Q2_touch = m2_touch >> Q2_body(m2_touch << Lines)

    # Separated by +0.8 along normal
    T_sep = (line_at_infinity.wedge(n_dir * 0.8) * -0.5).exp()
    m2_sep = T_sep * m2_touch
    Q2_sep = m2_sep >> Q2_body(m2_sep << Lines)

    # Overlapping by -0.6 along normal
    T_over = (line_at_infinity.wedge(n_dir * -0.6) * -0.5).exp()
    m2_over = T_over * m2_touch
    Q2_over = m2_over >> Q2_body(m2_over << Lines)

    # Solve contact parameters
    lam_sep, max_sep = find_contact_parameter(Q1, Q2_sep)
    lam_touch, max_touch = find_contact_parameter(Q1, Q2_touch)
    lam_over, max_over = find_contact_parameter(Q1, Q2_over)

    # State 1: Separated -> max det > 0
    assert max_sep > 0.5

    # State 2: Touching -> max det == 0 (to within numerical solver tolerance)
    assert np.isclose(max_touch, 0.0, atol=1e-8)

    # State 3: Overlapping -> max det < 0
    assert max_over < -0.5

    # Contact line and point verification for State 2
    L_star, p_star = extract_contact_line_and_point(Q1, Q2_touch, lam_touch)
    tang1 = L_star.regressive(Q1(L_star)).kernel.item()
    tang2 = (-L_star).regressive(Q2_touch(-L_star)).kernel.item()
    assert np.isclose(tang1, 0.0, atol=1e-6)
    assert np.isclose(tang2, 0.0, atol=1e-6)

    # Contact points on both ellipses must coincide
    p1 = normalize_point(Q1(L_star))
    p2 = normalize_point(Q2_touch(-L_star))
    np.testing.assert_allclose(p1.kernel, p2.kernel, atol=1e-5)
    np.testing.assert_allclose(p1.kernel, p_target.kernel, atol=1e-5)

