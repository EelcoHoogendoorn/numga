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
    Line,
    Point,
    Polarity,
    Pole,
    arccos,
    arccosh,
    draw_level_set,
    draw_line,
    draw_points,
    euclidean,
    mv,
    new_figure,
    point,
    style_axis,
)

BOX = (-1.2, 1.9, -1.2, 1.6)


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
    def cosh_distance(A: Point, B: Point):
        return -A.regressive(C(B)) / (A.regressive(C(A)) * B.regressive(C(B))).square_root()

    def cos_angle(l: Line, m: Line):
        return l.regressive(Q(m)) / (l.regressive(Q(l)) * m.regressive(Q(m))).square_root()

    # A triangle: its sides are joins of consecutive vertices, batched, and the angle at
    # each vertex is between the two sides leaving it. Gauss-Bonnet gives the area as the
    # angle defect.
    vertices = point(np.array([[0.0, 0.0], [0.65, 0.0], [0.2, 0.55]]))
    to_next = vertices.regressive(vertices[[1, 2, 0]])
    to_prev = vertices.regressive(vertices[[2, 0, 1]])
    angles = arccos(cos_angle(to_next, to_prev))
    sides = arccosh(cosh_distance(vertices, vertices[[1, 2, 0]]))
    area = np.pi - angles.sum()
    assert area > 0.0

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
    np.testing.assert_allclose(side.regressive(Q(perpendicular)).kernel, 0.0, atol=1e-12)
    np.testing.assert_allclose(foot.regressive(side).kernel, 0.0, atol=1e-12)
    np.testing.assert_allclose((cosh_distance(P, foot) - cosh_distance(reflected, foot)).kernel, 0.0, atol=1e-12)

    # -----------------------------------------------------------------------
    # 4. Circles are quadrics
    # -----------------------------------------------------------------------
    # Fixing the distance to a centre fixes the invariant, and clearing the square root
    # turns that into a quadric in P: the dyad of the centre's polar minus cosh²R times the
    # centre's self-invariant times the absolute. One expression, batched over the radii,
    # drawn as the level set P ∨ circle(P) = 0.
    radii = mv.scalar(np.cosh(np.array([0.25, 0.55, 0.9, 1.3]))[:, None] ** 2)
    centres = point(np.array([[0.0, 0.0], [0.45, 0.25]]))
    circles = []
    for centre in centres:
        polar = C(centre)
        circles.append(polar * polar.regressive(Point) - centre.regressive(C(centre)) * radii * C)

    # -----------------------------------------------------------------------
    # 5. Draw
    # -----------------------------------------------------------------------
    fig, (left, right) = new_figure()
    draw_level_set(left, C, BOX, colors="black", linewidths=2.0)
    for line in (to_next[0], to_next[1], to_next[2]):
        draw_line(left, line, BOX, color="#2563eb", linewidth=2.0)
    draw_line(left, perpendicular, BOX, color="#dc2626", linewidth=1.6)
    draw_points(left, vertices, color="#1d4ed8", s=40)
    draw_points(left, P, color="#dc2626", s=50, label="P")
    draw_points(left, foot, color="#b91c1c", marker="s", s=40, label="foot")
    draw_points(left, reflected, color="#f97316", s=45, label="reflection")
    draw_points(left, pole, color="#a855f7", marker="D", s=45, label="pole of BC")
    for vertex, theta in zip(vertices, angles):
        left.annotate(f"{np.degrees(theta):.1f}°", euclidean(vertex), textcoords="offset points", xytext=(6, 6), fontsize=9)
    style_axis(left, f"Triangle area {area:.3f} by Gauss-Bonnet; the perpendicular runs through the pole", BOX)
    left.legend(loc="lower left", fontsize=8)

    draw_level_set(right, C, BOX, colors="black", linewidths=2.0)
    for family, colour in zip(circles, ("#3b82f6", "#f97316")):
        for k in range(family.shape[0]):
            draw_level_set(right, family[k], BOX, colors=colour, linewidths=1.4)
    draw_points(right, centres, color="#111827", s=35)
    style_axis(right, "Circles as level sets of a quadric", BOX)

    plt.tight_layout()
    if plot_path:
        plt.savefig(plot_path, bbox_inches="tight")
        print(f"Figure saved to {plot_path}")
    return fig


if __name__ == "__main__":
    main()
    plt.show()
