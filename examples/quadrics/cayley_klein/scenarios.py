"""The hyperbolic plane built from its absolute: a triangle, a perpendicular and circles."""

from __future__ import annotations

import numpy as np

from examples.quadrics.cayley_klein.core import (
    Point, Polarity, circles, invariant, mv, perpendicular, point, triangle,
)


def hyperbolic_plane():
    """Build the hyperbolic plane from its absolute and construct a triangle, a perpendicular and circles."""
    # A line l paired with a point P through l & P is a rank-one map from points to lines.
    # Summing the three coordinate lines with signs gives the polarity of the conic
    # x**2 + y**2 - w**2 == 0, the unit circle. The pole map is the inverse polarity.
    C: Polarity = mv.x * mv.x.regressive(Point) + mv.y * mv.y.regressive(Point) - mv.w * mv.w.regressive(Point)

    vertices = point(np.array([[0.0, 0.0], [0.65, 0.0], [0.2, 0.55]]))
    sides, angles, lengths, area = triangle(C, vertices)

    P = point(np.array([-0.15, 0.15]))
    side = sides[1]
    pole, normal, foot, reflected = perpendicular(C, side, P)

    centres = point(np.array([[0.0, 0.0], [0.45, 0.25]]))
    radii = np.array([0.25, 0.55, 0.9, 1.3])
    rings = circles(C, centres, radii)

    # --- checks ---------------------------------------------------------------------------
    # A hyperbolic angle defect, a perpendicular, and a reflection that is an isometry.
    distance_pairing = -Point.regressive(C(mv.rotor() >> Point))
    assert area.to_array() > 0.0
    np.testing.assert_allclose((side & C.inverse()(normal)).to_array(), 0.0, atol=1e-12)
    np.testing.assert_allclose(
        invariant(distance_pairing, P, foot).to_array(),
        invariant(distance_pairing, reflected, foot).to_array(), atol=1e-12,
    )
    return C, vertices, sides, angles, area, P, foot, normal, reflected, pole, rings, centres


if __name__ == "__main__":
    from examples.animation import save_figure
    from examples.quadrics.cayley_klein import render

    save_figure(render.draw_hyperbolic_plane(*hyperbolic_plane()), "cayley_klein")
