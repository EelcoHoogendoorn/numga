"""A spherical conic with its foci, its tangent great circles, its cone and its polhodes."""

from __future__ import annotations

import numpy as np

from examples.quadrics.spherical_quadrics.core import (
    Plane, Point, cone, geodesic, great_circles, mv, oval, polarity, sphere,
)


def spherical_conic():
    # In principal axes both the primal quadric C (Plane <= Point) and the dual quadric
    # Q (Point <= Plane) are diagonal. A rotor moves either by the same sandwich.
    eigenvalues = np.array([1.0, 0.4, -0.8])
    rotor = (mv.xy * 0.25 + mv.yz * 0.15).exp()
    C = rotor >> polarity(eigenvalues)(rotor << Point)
    Q = rotor >> polarity(eigenvalues).inverse()(rotor << Plane)

    # Points along the spherical oval and its foci, carried into the world frame.
    curve, foci, theta_a = oval(eigenvalues, np.linspace(0, 2 * np.pi, 200))
    P = rotor >> curve
    foci = rotor >> foci
    F1, F2 = foci

    # The distance between unit points on the sphere is arccos(-P · F). The sum of the
    # geodesic distances from F1 and F2 to any point on the oval is constant: 2 θa.
    dist_F1 = (-(P | F1)).clip(-1.0, 1.0).arccos()
    dist_F2 = (-(P | F2)).clip(-1.0, 1.0).arccos()
    focal_sum = dist_F1 + dist_F2

    # The polar plane of P with respect to C is its tangent great circle, and every tangent
    # great circle satisfies the dual equation π ∨ Q(π) = 0.
    tangents = C(P).normalized()

    # Dual focal property: the product of the sines of the distances from F1 and F2 to the
    # tangent great circles is constant; sin(dist(F, π)) = |F ∨ π|.
    dual_product = F1.regressive(tangents).norm() * F2.regressive(tangents).norm()

    # For the figure: geodesics from both foci to one point of the oval, sixteen tangent
    # great circles, the cone through the oval, and the potential P ∨ C(P) on the sphere.
    sample = P[25]
    arcs = geodesic(foci[:, None], sample, np.linspace(0, 1, 30))
    touching = np.linspace(0, P.shape[0] - 1, 16, dtype=int)
    circles = great_circles(tangents[touching][:, None], P[touching][:, None], np.linspace(0, 2 * np.pi, 120))
    radius, t = np.meshgrid(np.linspace(0.1, 1.25, 20), np.linspace(0, 2 * np.pi, 60))
    surface = rotor >> cone(eigenvalues, radius, t)
    grid = sphere(60, 30)
    potential = grid.regressive(C(grid))

    # --- checks ---------------------------------------------------------------------------
    np.testing.assert_allclose(P.regressive(C(P)).to_array(), 0.0, atol=1e-14)
    np.testing.assert_allclose(focal_sum.to_array(), 2.0 * theta_a, atol=1e-11)
    np.testing.assert_allclose(tangents.regressive(Q(tangents)).to_array(), 0.0, atol=1e-14)
    np.testing.assert_allclose(dual_product.to_array(), dual_product.mean().to_array(), atol=1e-14)
    return P, foci, arcs, sample, theta_a, tangents, tangents[touching], circles, surface, grid, potential


if __name__ == "__main__":
    from examples.animation import save_figure
    from examples.quadrics.spherical_quadrics import render

    save_figure(render.draw_spherical_conic(*spherical_conic()), "spherical_quadrics")
