"""Two ellipses posed apart, touching and overlapping, classified by the blends of their dual quadrics."""

from __future__ import annotations

import numpy as np

from examples.quadrics.quadric_collision.core import (
    Line, cubic_peak, ellipse, infinity, motor, mv, blend, tangent_line,
)


def collision():
    """A second ellipse touching the first along a chosen normal, then moved apart and into it."""
    # The first ellipse: semi-axes 2 and 1, turned 25° and centred at (-1.2, 0).
    placement = motor(-1.2, 0.0, np.radians(25))
    Q1 = placement >> ellipse(2.0, 1.0)(placement << Line)

    # The second ellipse, turned -35°, is translated so that its tangent point with the
    # opposite normal lands on the first ellipse's tangent point with a 22° normal.
    normal = (mv.x * np.cos(np.radians(22)) + mv.y * np.sin(np.radians(22))).normalized()
    shape = ellipse(1.6, 0.9)
    turn = motor(0.0, 0.0, np.radians(-35))
    turned = turn >> shape(turn << Line)
    target = Q1(tangent_line(Q1, normal))
    start = turned(tangent_line(turned, -normal))
    # Their displacement is the difference of the two points taken at unit weight.
    displacement = target / (target & infinity) - start / (start & infinity)
    touch = (infinity.wedge(displacement.dual()) * -0.5).exp() * turn

    # Three poses of it along the normal: 0.8 apart, touching, and 0.6 into the first.
    offsets = np.array([0.8, 0.0, -0.6])
    poses = (infinity.wedge(normal * offsets) * -0.5).exp() * touch
    Q2 = poses >> shape(poses << Line)

    # Four determinant samples determine the cubic determinant of the blend exactly.
    # Only locating its peak leaves the algebra.
    samples = mv.scalar([[0.0], [1.0], [2.0], [-1.0]])
    parameter, maximum = cubic_peak(blend(Q1, Q2, samples[:, None]).dual().det())

    # At contact the blend has a null line. Its singular vector is the shared tangent;
    # each quadric maps that tangent to the same contact point.
    _, _, lines = blend(Q1, Q2[1], parameter[1]).dual().svd()
    contact_line = lines[-1].normalized()
    contact_point = Q1(contact_line).normalized()

    # For the figure: the determinant along the blend, the tangent lines of both ellipses
    # with the chosen normal and their midline, and the blends between them.
    sweep = mv.scalar(np.linspace(0.001, 0.999, 500)[:, None])
    determinants = blend(Q1, Q2, sweep[:, None]).dual().det()
    first = tangent_line(Q1, normal)
    second = tangent_line(Q2, -normal)
    midline = ((first - second) * 0.5).normalized()
    # A hyperbola between the two.
    separating = blend(Q1, Q2[0], parameter[0]).inverse()
    deforming = blend(Q1, Q2[1], 0.45)
    bridging = blend(Q1, Q2[2], parameter[2])

    # --- checks ---------------------------------------------------------------------------
    # The poses are apart, touching and overlapping; the contact line is a common tangent, and
    # both ellipses map it to one contact point.
    assert maximum[0].to_array() > 0
    np.testing.assert_allclose(maximum[1].to_array(), 0, atol=1e-8)
    assert maximum[2].to_array() < 0
    np.testing.assert_allclose((contact_line & Q1(contact_line)).to_array(), 0, atol=1e-6)
    np.testing.assert_allclose((Q2[1](contact_line).normalized() & contact_point).norm().to_array(), 0, atol=1e-5)
    return (Q1, Q2, parameter, maximum, sweep, determinants, first, second, midline,
            separating, deforming, bridging, contact_line, contact_point)


if __name__ == "__main__":
    from examples.animation import save_figure
    from examples.quadrics.quadric_collision import render

    save_figure(render.draw_collision(*collision()), "quadric_collision_2d")
