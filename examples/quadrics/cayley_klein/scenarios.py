"""The hyperbolic space built from its absolute: a triangle, a perpendicular and spheres."""

from __future__ import annotations

import numpy as np

from numga import stack
from examples.quadrics.cayley_klein.core import (
    Point, Polarity, spheres, invariant, mv, perpendicular, triangle,
)

# The triangle's vertices, the point a perpendicular is dropped from, and the spheres' centres and radii.
VERTICES = [[0, 0], [0.65, 0], [0.2, 0.55]]
PROBE = [-0.15, 0.15]
CENTRES = [[0, 0], [0.45, 0.25]]
RADII = np.array([0.25, 0.55, 0.9, 1.3])
# The signs of the coordinate planes x, y and w in the absolute, the quadric x**2 + y**2 - w**2 == 0: the unit sphere.
SIGNS = np.array([1.0, 1.0, -1.0])


# --- math -----------------------------------------------------------------------------
def hyperbolic_space():
    """The hyperbolic space built from its absolute, and in it a triangle, a perpendicular and spheres: the absolute,
    the triangle's vertices, sides and angles, the probe point, the foot of its perpendicular to the second side,
    the perpendicular, the point reflected, the side's pole, the spheres and their centres."""
    # A plane paired with a point is a rank-one map from points to planes. Summing the three coordinate planes
    # with signs gives the polarity of the quadric, the absolute. The pole map is the inverse polarity.
    coordinate_planes = stack([mv.x, mv.y, mv.w])                              # [3] Plane+
    absolute = (coordinate_planes * (coordinate_planes & Point) * SIGNS).sum(axis=0)   # [] Plane+ <- Point-

    vertices = mv.point(VERTICES)                                              # [3] Point-
    sides, angles, _, _ = triangle(absolute, vertices)

    probe = mv.point(PROBE)                                                    # [] Point-
    pole, normal, foot, reflected = perpendicular(absolute, sides[1], probe)

    centres = mv.point(CENTRES)                                                # [centres] Point-
    rings = spheres(absolute, centres, RADII)                                  # [centres, radii] Plane+ <- Point-
    return absolute, vertices, sides, angles, probe, foot, normal, reflected, pole, rings, centres


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_figure
    from examples.quadrics.cayley_klein import render

    scene = hyperbolic_space()
    save_figure(render.draw_hyperbolic_space(*scene), "cayley_klein")

    # checks
    # A hyperbolic angle defect, a perpendicular, and a reflection that is an isometry: the perpendicular's
    # pole, the meet of the polars of the two points it joins, lies on the side, and the point and its
    # reflection are equally far from the foot.
    absolute, vertices, sides, _, probe, foot, _, reflected, pole, _, _ = scene
    _, _, _, area = triangle(absolute, vertices)
    distance_pairing = -(Point & absolute(Point))
    assert area > 0.0
    np.testing.assert_allclose(((absolute(probe) ^ absolute(pole)) & sides[1]).to_array(), 0.0, atol=1e-12)
    np.testing.assert_allclose(invariant(distance_pairing, probe, foot).to_array(),
                               invariant(distance_pairing, reflected, foot).to_array(), atol=1e-12)


if __name__ == "__main__":
    main()
