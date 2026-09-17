"""Cayley-Klein geometry in PGA2D: the metric is a quadric you choose.

Projective geometry has joins and meets but no distances. Pick one conic, the absolute,
as a polarity map C from points to lines, and every metric notion follows from C and its
inverse Q alone: distances and angles are cross ratios against the absolute, the
perpendiculars to a line all pass through its pole Q(l), and a circle is the quadric of
points at fixed cross ratio from its centre. With the unit circle as the absolute this is
the hyperbolic Beltrami-Klein disk. Flip one sign and the same lines do elliptic geometry.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples import PLOT_DIR
from examples.quadrics.cayley_klein_plumbing import (
    draw_geometry,
    Line,
    Point,
    Polarity,
    Pole,
    mv,
    point,
)


def main(plot_path: str = str(PLOT_DIR / "cayley_klein.png")) -> plt.Figure:
    """Build the hyperbolic plane from its absolute and draw a triangle, a perpendicular and circles."""

    # -----------------------------------------------------------------------
    # 1. The absolute: a polarity built from line dyads
    # -----------------------------------------------------------------------
    # A line l paired with a point through l ∨ P is a rank-one map from points to lines.
    # Summing the three coordinate lines with signs gives the polarity of the conic
    # x² + y² - w² = 0, the unit circle. The pole map is the inverse polarity.
    C: Polarity = mv.x * mv.x.regressive(Point) + mv.y * mv.y.regressive(Point) - mv.w * mv.w.regressive(Point)
    Q: Pole = C.inverse()

    # -----------------------------------------------------------------------
    # 2. Distances and angles are cross ratios against the absolute
    # -----------------------------------------------------------------------
    # The invariant of two points is (P₁ ∨ C(P₂)) / √((P₁ ∨ C(P₁))(P₂ ∨ C(P₂))), and Cayley's
    # distance is its arccosh. The invariant of two lines is the same with Q, and the angle
    # is its arccos. Both stay inside the algebra until the very last step.
    distance_pairing = -Point.regressive(C(mv.rotor() >> Point))
    angle_pairing = Line.regressive(Q(mv.rotor() >> Line))

    # A triangle: its sides are joins of consecutive vertices, batched, and the angle at
    # each vertex is between the two sides leaving it. Gauss-Bonnet gives the area as the
    # angle defect.
    vertices = point(np.array([[0.0, 0.0], [0.65, 0.0], [0.2, 0.55]]))
    to_next = vertices.regressive(vertices[[1, 2, 0]])
    to_prev = vertices.regressive(vertices[[2, 0, 1]])
    angles = (angle_pairing(to_next, to_prev) /
              (angle_pairing(to_next, to_next) * angle_pairing(to_prev, to_prev)).square_root()).clip(-1.0, 1.0).arccos()
    neighbours = vertices[[1, 2, 0]]
    sides = (distance_pairing(vertices, neighbours) /
             (distance_pairing(vertices, vertices) * distance_pairing(neighbours, neighbours)).square_root()).clip(1.0, np.inf).arccosh()
    area = np.pi - angles.sum()

    # -----------------------------------------------------------------------
    # 3. Perpendiculars and reflections come from the pole
    # -----------------------------------------------------------------------
    # Every line perpendicular to l passes through its pole Q(l), so the perpendicular from
    # P is the join P ∨ Q(l) and the foot is its meet with l. Reflection across l is the
    # harmonic homology centred on the pole, and it preserves the distance to the foot.
    P = point(np.array([-0.15, 0.15]))
    side = to_next[1]
    pole = Q(side)
    perpendicular = P.regressive(pole)
    foot = side.wedge(perpendicular)
    reflected = P - pole * (2.0 * P.regressive(side) / pole.regressive(side))

    # -----------------------------------------------------------------------
    # 4. Circles are quadrics
    # -----------------------------------------------------------------------
    # Fixing the distance to a centre fixes the invariant, and clearing the square root
    # turns that into a quadric in P: the dyad of the centre's polar minus cosh²R times the
    # centre's self-invariant times the absolute. One expression, batched over the radii,
    # drawn as the level set P ∨ circle(P) = 0.
    radii = mv.scalar(np.cosh(np.array([0.25, 0.55, 0.9, 1.3]))[:, None] ** 2)
    centres = point(np.array([[0.0, 0.0], [0.45, 0.25]]))
    centre = centres[:, None]
    polar = C(centre)
    circles = polar * polar.regressive(Point) - centre.regressive(polar) * radii * C

    # -----------------------------------------------------------------------
    # 5. Draw
    # -----------------------------------------------------------------------
    fig = draw_geometry(C, to_next, perpendicular, vertices, P, foot, reflected, pole, angles, area, circles, centres, plot_path)


    # --- checks -------------------------------------------------------------
    assert area.kernel.item() > 0.0
    np.testing.assert_allclose(side.regressive(Q(perpendicular)).kernel, 0.0, atol=1e-12)
    np.testing.assert_allclose(foot.regressive(side).kernel, 0.0, atol=1e-12)
    np.testing.assert_allclose(
        (distance_pairing(P, foot) / (distance_pairing(P, P) * distance_pairing(foot, foot)).square_root()).kernel,
        (distance_pairing(reflected, foot) / (distance_pairing(reflected, reflected) * distance_pairing(foot, foot)).square_root()).kernel,
        atol=1e-12,
    )

    return fig


if __name__ == "__main__":
    main()
    plt.show()
