"""Unit tests for Cayley-Klein geometry: the absolute as a polarity and what follows from it."""

from __future__ import annotations

import numpy as np

from examples.quadrics.cayley_klein import main
from examples.quadrics.cayley_klein_plumbing import Point, Polarity, Pole, mv, point

HYPERBOLIC = mv.x * mv.x.regressive(Point) + mv.y * mv.y.regressive(Point) - mv.w * mv.w.regressive(Point)
ELLIPTIC = mv.x * mv.x.regressive(Point) + mv.y * mv.y.regressive(Point) + mv.w * mv.w.regressive(Point)


def invariant(C, A: Point, B: Point):
    return A.regressive(C(B)) / (A.regressive(C(A)) * B.regressive(C(B))).square_root()


def test_absolute_from_line_dyads_is_the_diagonal_polarity():
    assert HYPERBOLIC.gatype == Polarity
    np.testing.assert_allclose(HYPERBOLIC.kernel, np.diag([1.0, 1.0, -1.0]))
    Q = HYPERBOLIC.inverse()
    assert Q.gatype == Pole
    np.testing.assert_allclose(Q.kernel, np.diag([1.0, 1.0, -1.0]), atol=1e-14)
    np.testing.assert_allclose(ELLIPTIC.kernel, np.eye(3))


def test_hyperbolic_distance_along_a_diameter_is_arctanh():
    origin = point(np.array([0.0, 0.0]))
    for x in (0.2, 0.5, 0.75):
        np.testing.assert_allclose((-invariant(HYPERBOLIC, origin, point(np.array([x, 0.0])))).clip(1.0, np.inf).arccosh().to_array(), np.arctanh(x), atol=1e-12)


def test_elliptic_distance_along_a_diameter_is_arctan():
    """On the gnomonic chart of the sphere, the distance from the origin is arctan of the radius."""
    origin = point(np.array([0.0, 0.0]))
    for x in (0.2, 0.5, 2.0):
        np.testing.assert_allclose(invariant(ELLIPTIC, origin, point(np.array([x, 0.0]))).clip(-1.0, 1.0).arccos().to_array(), np.arctan(x), atol=1e-12)


def test_perpendicular_through_the_pole_and_reflection_in_both_geometries():
    for C in (HYPERBOLIC, ELLIPTIC):
        Q = C.inverse()
        side = point(np.array([0.65, 0.0])).regressive(point(np.array([0.2, 0.55])))
        P = point(np.array([-0.15, 0.15]))
        pole = Q(side)
        perpendicular = P.regressive(pole)
        foot = side.wedge(perpendicular)
        reflected = P - pole * (2.0 * P.regressive(side) / pole.regressive(side))
        np.testing.assert_allclose(side.regressive(Q(perpendicular)).kernel, 0.0, atol=1e-14)
        np.testing.assert_allclose(foot.regressive(side).kernel, 0.0, atol=1e-14)
        np.testing.assert_allclose((invariant(C, P, foot) - invariant(C, reflected, foot)).kernel, 0.0, atol=1e-12)
        np.testing.assert_allclose(reflected.regressive(perpendicular).kernel, 0.0, atol=1e-14)


def test_triangle_angle_sum_is_below_pi_hyperbolic_and_above_pi_elliptic():
    vertices = point(np.array([[0.0, 0.0], [0.65, 0.0], [0.2, 0.55]]))
    to_next = vertices.regressive(vertices[[1, 2, 0]])
    to_prev = vertices.regressive(vertices[[2, 0, 1]])
    sums = {}
    for name, C in (("hyperbolic", HYPERBOLIC), ("elliptic", ELLIPTIC)):
        Q = C.inverse()
        sums[name] = (to_next.regressive(Q(to_prev)) / (to_next.regressive(Q(to_next)) * to_prev.regressive(Q(to_prev))).square_root()).clip(-1.0, 1.0).arccos().sum().to_array()
    assert sums["hyperbolic"] < np.pi < sums["elliptic"]


def test_circle_quadric_contains_the_points_at_its_radius():
    C = HYPERBOLIC
    centre, R = point(np.array([0.45, 0.25])), 0.9
    polar = C(centre)
    circle = polar * polar.regressive(Point) - centre.regressive(C(centre)) * (np.cosh(R) ** 2) * C
    lo, hi = 0.45, 0.999
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if (-invariant(C, centre, point(np.array([mid, 0.25])))).clip(1.0, np.inf).arccosh() < R else (lo, mid)
    on_circle = point(np.array([lo, 0.25]))
    np.testing.assert_allclose(on_circle.regressive(circle(on_circle)).kernel, 0.0, atol=1e-12)


def test_tutorial_runs_and_saves(tmp_path):
    out = tmp_path / "cayley_klein.png"
    main(plot_path=str(out))
    assert out.exists()
