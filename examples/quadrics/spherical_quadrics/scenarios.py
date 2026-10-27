"""A spherical conic with its foci, its tangent great circles, its cone and its polhodes."""

from __future__ import annotations

import numpy as np

from examples.quadrics.spherical_quadrics.core import Plane, Point, mv, oval, polarity

# The polarity's eigenvalues on the planes x, y and z, and the turn that places it, by its two planes.
EIGENVALUES = np.array([1.0, 0.4, -0.8])
TURN_XY, TURN_YZ = 0.25, 0.15
# Points along the oval, and the one the geodesics from the foci run to.
N_CURVE = 200
SAMPLE = 25


# --- math -----------------------------------------------------------------------------
def spherical_conic():
    """The oval comes with its foci and one point of it, its semi-major arc, its tangent great circles, and the
    polarity and its dual."""
    # In principal axes both the polarity, from points to planes, and its dual, from planes to points,
    # are diagonal; a rotor moves either by the same sandwich.
    rotor = (mv.xy * TURN_XY + mv.yz * TURN_YZ).exp()                          # [] Rotor+
    principal = polarity(EIGENVALUES)                                          # [] Plane+ <- Point-
    conic = rotor >> principal(rotor << Point)                                 # [] Plane+ <- Point-
    dual = rotor >> principal.solve(rotor << Plane)                            # [] Point- <- Plane+

    # Points along the spherical oval and its foci, carried into the world.
    unturned, unturned_foci, semi_major = oval(EIGENVALUES, np.linspace(0, 2 * np.pi, N_CURVE))
    curve = rotor >> unturned                                                  # [curve] Point-
    foci = rotor >> unturned_foci                                              # [2] Point-

    # The polar plane of a point of the oval is its tangent great circle.
    tangents = conic(curve).normalized()                                       # [curve] Plane+

    # For the figure: one point of the oval, which the geodesics from both foci run to.
    sample = curve[SAMPLE]
    return curve, foci, sample, semi_major, tangents, conic, dual


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_figure
    from examples.quadrics.spherical_quadrics import render

    curve, foci, sample, semi_major, tangents, conic, dual = spherical_conic()
    save_figure(render.draw_spherical_conic(curve, foci, sample, tangents, conic), "spherical_quadrics")

    # checks
    # Every point of the oval lies on its own polar. The distance between unit points on the sphere is
    # `curve.dot(focus).arccos()`, and the sum of the distances to both foci is twice the semi-major arc.
    # Every tangent great circle lies on the dual conic, and the product of the sines of the foci's distances
    # to the tangent circles, each the norm of the focus joined with the circle, is constant.
    first, second = foci
    focal_sum = curve.dot(first).clip(-1.0, 1.0).arccos() + curve.dot(second).clip(-1.0, 1.0).arccos()
    dual_product = (first & tangents).norm() * (second & tangents).norm()
    np.testing.assert_allclose((curve & conic(curve)).to_array(), 0.0, atol=1e-12)
    np.testing.assert_allclose(focal_sum.to_array(), 2.0 * semi_major, atol=1e-11)
    np.testing.assert_allclose((tangents & dual(tangents)).to_array(), 0.0, atol=1e-12)
    np.testing.assert_allclose(dual_product.to_array(), dual_product.mean().to_array(), atol=1e-14)


if __name__ == "__main__":
    main()
