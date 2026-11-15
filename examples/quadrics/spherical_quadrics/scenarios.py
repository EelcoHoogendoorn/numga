"""Scene setup and verification for a spherical conic, its foci, its tangent envelope, and its polhodes."""

from __future__ import annotations

import numpy as np

from examples.quadrics.spherical_quadrics.core import (
    DualPolarity,
    Plane,
    Point,
    Polarity,
    Scalar,
    conic_locus,
    mv,
    principal_polarity,
)

# Principal eigenvalues on the coordinate planes x, y, z and the rotor bivector angles that place the conic.
eigenvalues = np.array([1.0, 0.4, -0.8])
turn_xy, turn_yz = 0.25, 0.15
# Sample count along the spherical oval and index of the point reached by the focal geodesics.
n_oval = 200
sample_index = 25


# --- math -----------------------------------------------------------------------------
def spherical_conic_scene() -> tuple[Point, Point, Point, Scalar, Plane, Polarity, DualPolarity]:
    """Places a diagonal spherical polarity by a rotor and collects its oval (`[n_oval] Point-`), foci (`[2] Point-`), sample point, semi-major arc, tangent planes, polarity, and dual polarity."""
    rotor = (mv.xy * turn_xy + mv.yz * turn_yz).exp()                           # [] Rotor+
    principal = principal_polarity(eigenvalues)                                 # [] Plane+ <- Point-
    conic = rotor >> principal(rotor << Point)                                  # [] Plane+ <- Point-
    dual = rotor >> principal.solve(rotor << Plane)                             # [] Point- <- Plane+

    rest_oval, rest_foci, semi_major = conic_locus(principal, np.linspace(0, 2 * np.pi, n_oval))
    oval_points = (rotor >> rest_oval).retyped(Point)                           # [n_oval] Point-
    foci = (rotor >> rest_foci).retyped(Point)                                  # [2] Point-

    # The polar plane of each point on the oval is its tangent great circle.
    tangents = conic(oval_points).normalized()                                  # [n_oval] Plane+
    sample_point = oval_points[sample_index]                                    # [] Point-
    return oval_points, foci, sample_point, semi_major, tangents, conic, dual


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_figure
    from examples.quadrics.spherical_quadrics import render

    oval_points, foci, sample_point, semi_major, tangents, conic, dual = spherical_conic_scene()
    save_figure(render.conic_figure(oval_points, foci, sample_point, tangents, conic), "spherical_quadrics")

    # checks
    # Every point of the oval lies on its own polar, the geodesic distances to both foci sum to twice the
    # semi-major arc, every tangent great circle lies on the dual conic, and the product of the sines of the
    # foci's distances to the tangent circles is constant.
    focal_sum = (oval_points[:, None].reverse() | foci).clip(-1.0, 1.0).arccos().sum(axis=-1)  # [n_oval] Scalar+
    dual_product = (tangents[:, None] & foci).abs().cumprod(axis=-1)[:, -1]     # [n_oval] Scalar+
    assert ((conic(oval_points) & oval_points).abs() < 1e-12).all()
    assert ((focal_sum - 2 * semi_major).abs() < 1e-11).all()
    assert ((tangents & dual(tangents)).abs() < 1e-12).all()
    assert ((dual_product - dual_product.mean()).abs() < 1e-13).all()


if __name__ == "__main__":
    main()
