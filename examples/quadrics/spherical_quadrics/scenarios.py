"""A spherical conic with its foci, its tangent great circles, its cone and its polhodes."""

from __future__ import annotations

import numpy as np

from examples.quadrics.spherical_quadrics.core import (
    Plane, Point, Turn, cone, geodesic, great_circles, mv, oval, polarity, sphere,
)

# The polarity's eigenvalues on the planes x, y and z, and the turn that places it, by its two planes.
EIGENVALUES = np.array([1.0, 0.4, -0.8])
TURN_XY, TURN_YZ = 0.25, 0.15
# Points along the oval; the one the geodesics from the foci run to, and their samples; the tangent great
# circles drawn and their samples; the cone's rings and samples around; the sphere's grid.
N_CURVE = 200
SAMPLE = 25
N_ARC = 30
N_TOUCHING = 16
N_CIRCLE = 120
CONE_RADII = np.linspace(0.1, 1.25, 20)
N_AROUND = 60
GRID = (120, 60)


# --- math -----------------------------------------------------------------------------
def spherical_conic():
    """The oval, its foci and their geodesics to one point of it, its semi-major arc, its tangent great
    circles, the cone through it, the sphere's grid with the polarity's form on it, and the polarity and
    its dual."""
    # In principal axes both the polarity, from points to planes, and its dual, from planes to points,
    # are diagonal; a rotor moves either by the same sandwich.
    rotor = (mv.xy * TURN_XY + mv.yz * TURN_YZ).exp()                          # [] Rotor
    conic = rotor >> polarity(EIGENVALUES)(rotor << Point)                     # [] Plane <- Point
    dual = rotor >> polarity(EIGENVALUES).solve(rotor << Plane)                # [] Point <- Plane

    # Points along the spherical oval and its foci, carried into the world.
    unturned, unturned_foci, semi_major = oval(EIGENVALUES, np.linspace(0, 2 * np.pi, N_CURVE))
    curve = rotor >> unturned                                                  # [curve] Point
    foci = rotor >> unturned_foci                                              # [2] Point

    # The polar plane of a point of the oval is its tangent great circle.
    tangents = conic(curve).normalized()                                       # [curve] Plane

    # For the figure: geodesics from both foci to one point of the oval, some tangent great circles, the
    # cone through the oval, and the polarity's form on the sphere.
    sample = curve[SAMPLE]
    arcs = geodesic(foci[:, None], sample, np.linspace(0, 1, N_ARC))           # [2, arc] Point
    touching = np.linspace(0, N_CURVE - 1, N_TOUCHING, dtype=int)
    turns = mv(Turn, np.linspace(0, 2 * np.pi, N_CIRCLE)[:, None])             # [circle] Turn
    circles = great_circles(tangents[touching][:, None], curve[touching][:, None], turns)   # [touching, circle] Point
    radius, around = np.meshgrid(CONE_RADII, np.linspace(0, 2 * np.pi, N_AROUND))
    surface = rotor >> cone(EIGENVALUES, radius, around)                       # [around, radii] Point
    grid = sphere(*GRID)                                                       # [latitudes, longitudes] Point
    potential = grid & conic(grid)                                             # [latitudes, longitudes] Scalar
    return curve, foci, arcs, sample, semi_major, tangents, tangents[touching], circles, surface, grid, potential, conic, dual


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_figure
    from examples.quadrics.spherical_quadrics import render

    curve, foci, arcs, sample, semi_major, tangents, touching, circles, surface, grid, potential, conic, dual = spherical_conic()
    save_figure(render.draw_spherical_conic(curve, foci, arcs, sample, tangents, touching, circles, surface, grid,
                                            potential, conic), "spherical_quadrics")

    # checks
    # Every point of the oval lies on its own polar. The distance between unit points on the sphere is
    # `(-(curve | focus)).arccos()`, and the sum of the distances to both foci is twice the semi-major arc.
    # Every tangent great circle lies on the dual conic, and the product of the sines of the foci's distances
    # to the tangent circles, each the norm of the focus joined with the circle, is constant.
    first, second = foci
    focal_sum = (-(curve | first)).clip(-1.0, 1.0).arccos() + (-(curve | second)).clip(-1.0, 1.0).arccos()
    dual_product = (first & tangents).norm() * (second & tangents).norm()
    np.testing.assert_allclose((curve & conic(curve)).to_array(), 0.0, atol=1e-12)
    np.testing.assert_allclose(focal_sum.to_array(), 2.0 * semi_major, atol=1e-11)
    np.testing.assert_allclose((tangents & dual(tangents)).to_array(), 0.0, atol=1e-12)
    np.testing.assert_allclose(dual_product.to_array(), dual_product.mean().to_array(), atol=1e-14)


if __name__ == "__main__":
    main()
