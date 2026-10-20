"""Two ellipses posed apart, touching and overlapping, classified by the blends of their dual quadrics."""

from __future__ import annotations

import numpy as np

from examples.quadrics.quadric_collision.core import (
    Line, blend, cubic_peak, ellipse, infinity, motor, mv, tangent_line,
)

# The first ellipse: its semi-axes, its centre and its turn.
FIRST_SEMI = (2.0, 1.0)
FIRST_CENTRE = (-1.2, 0.0)
FIRST_TURN = np.radians(25)
# The second ellipse: its semi-axes and its turn; it touches the first where the first's normal is turned
# this far from x, and is posed apart, touching and overlapping by these offsets along that normal.
SECOND_SEMI = (1.6, 0.9)
SECOND_TURN = np.radians(-35)
CONTACT_NORMAL = np.radians(22)
OFFSETS = np.array([0.8, 0.0, -0.6])
# The shares at which the blend's determinant, a cubic, is sampled to fix it; the shares it is drawn at;
# and the share of the blended ellipse drawn between the touching pair.
SAMPLES = [[0.0], [1.0], [2.0], [-1.0]]
N_SHARES = 500
DRAWN_SHARE = 0.45


# --- math -----------------------------------------------------------------------------
def collision():
    """A second ellipse touching the first along a chosen normal, then moved apart and into it."""
    placement = motor(*FIRST_CENTRE, FIRST_TURN)
    fixed = placement >> ellipse(*FIRST_SEMI)(placement << Line)               # [] Point <- Line

    # The second ellipse, turned, is translated so that its tangent point with the opposite normal lands
    # on the first ellipse's tangent point with the chosen normal.
    normal = (mv.xy * (-CONTACT_NORMAL / 2)).exp() >> mv.x                     # [] Line
    shape = ellipse(*SECOND_SEMI)                                              # [] Point <- Line
    turn = motor(0.0, 0.0, SECOND_TURN)
    turned = turn >> shape(turn << Line)                                       # [] Point <- Line
    target = fixed(tangent_line(fixed, normal))                                # [] Point
    start = turned(tangent_line(turned, -normal))                              # [] Point
    # Their displacement is the difference of the two points taken at unit weight.
    displacement = target / (target & infinity) - start / (start & infinity)   # [] Point
    touch = (infinity.wedge(displacement.dual()) * -0.5).exp() * turn          # [] Motor
    # The poses along the normal: apart, touching, and into the first.
    poses = (infinity.wedge(normal * OFFSETS) * -0.5).exp() * touch            # [poses] Motor
    moved = poses >> shape(poses << Line)                                      # [poses] Point <- Line

    # Four samples of the blend's determinant fix the cubic exactly; only locating its peak leaves the
    # algebra.
    parameter, maximum = cubic_peak(blend(fixed, moved, mv.scalar(SAMPLES)[:, None]).dual().det())   # [poses] Scalar each

    # At contact the blend at the peak has a null line, its last singular vector: the shared tangent, which
    # each ellipse maps to the same contact point.
    _, _, lines = blend(fixed, moved[1], parameter[1]).dual().svd()
    contact_line = lines[-1].normalized()                                      # [] Line
    contact_point = fixed(contact_line).normalized()                           # [] Point

    # For the figure: the determinant along the blend, the tangent lines of both ellipses with the chosen
    # normal and their midline, and the blends between them: a hyperbola between the pair apart, an
    # ellipse between the pair touching, and one bridging the pair overlapping.
    shares = mv.scalar(np.linspace(0.001, 0.999, N_SHARES)[:, None])           # [shares] Scalar
    determinants = blend(fixed, moved, shares[:, None]).dual().det()           # [shares, poses] Scalar
    first = tangent_line(fixed, normal)                                        # [] Line
    second = tangent_line(moved, -normal)                                      # [poses] Line
    midline = ((first - second) * 0.5).normalized()                            # [poses] Line
    separating = blend(fixed, moved[0], parameter[0]).inverse()                # [] Line <- Point
    deforming = blend(fixed, moved[1], DRAWN_SHARE)                            # [] Point <- Line
    bridging = blend(fixed, moved[2], parameter[2])                            # [] Point <- Line
    return (fixed, moved, parameter, maximum, shares, determinants, first, second, midline,
            separating, deforming, bridging, contact_line, contact_point)


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.animation import save_figure
    from examples.quadrics.quadric_collision import render

    scene = collision()
    save_figure(render.draw_collision(*scene), "quadric_collision_2d")

    # checks
    # The poses are apart, touching and overlapping, by the sign of the blend's peak determinant; the
    # contact line is a common tangent, and both ellipses map it to one contact point.
    fixed, moved, _, maximum = scene[:4]
    contact_line, contact_point = scene[-2:]
    assert maximum[0].to_array() > 0 and maximum[2].to_array() < 0
    np.testing.assert_allclose(maximum[1].to_array(), 0, atol=1e-12)
    np.testing.assert_allclose((contact_line & fixed(contact_line)).to_array(), 0, atol=1e-12)
    np.testing.assert_allclose((moved[1](contact_line).normalized() & contact_point).norm().to_array(), 0, atol=1e-12)


if __name__ == "__main__":
    main()
