"""Two ellipses posed apart, touching and overlapping, classified by the blends of their dual quadrics, and the second
sliding along the normal through contact with the first."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from examples.quadrics.quadric_collision.core import (
    Line, Polarity, Quadric, Scalar, blend, cubic_peak, ellipse, infinity, motor, mv, tangent_line,
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
# The offsets the second ellipse slides through, from apart into the first, and the duration of a frame.
SLIDE = np.linspace(0.8, -0.6, 57)
DURATION_MS = 60
# The shares at which the blend's determinant, a cubic, is sampled to fix it, and the shares it is drawn at.
SAMPLES = [[0.0], [1.0], [2.0], [-1.0]]
N_SHARES = 500


# --- math -----------------------------------------------------------------------------
def pair():
    """The first ellipse, the second at its own centre, the first's normal at the contact, and the motion that
    lays the second against the first along that normal."""
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
    return fixed, shape, normal, touch


def collision():
    """A second ellipse touching the first along a chosen normal, then moved apart and into it."""
    fixed, shape, normal, touch = pair()
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
    return fixed, moved, parameter, maximum, contact_line, contact_point


def slide(offsets: np.ndarray) -> tuple[Quadric, Line, Scalar, Iterator[tuple]]:
    """The first ellipse, its tangent line with the chosen normal, the shares the determinant is drawn at, and the
    second ellipse at each offset along the normal from touching, with what the figure draws of it."""
    fixed, shape, normal, touch = pair()
    first = tangent_line(fixed, normal)                                        # [] Line
    shares = mv.scalar(np.linspace(0.001, 0.999, N_SHARES)[:, None])           # [shares] Scalar

    def moments() -> Iterator[tuple[Quadric, Line, Line, Polarity, Scalar, Scalar, Scalar]]:
        # Each moment: the second ellipse, its tangent line with the opposite normal and the midline of the two
        # tangents, the blend at the determinant's peak as a map from points to their polar lines, and the
        # determinant along the blends with its peak.
        for offset in offsets:
            pose = (infinity.wedge(normal * offset) * -0.5).exp() * touch      # [] Motor
            moved = pose >> shape(pose << Line)                                # [] Point <- Line
            parameter, maximum = cubic_peak(blend(fixed, moved, mv.scalar(SAMPLES)).dual().det())   # [] Scalar each
            second = tangent_line(moved, -normal)                              # [] Line
            midline = ((first - second) * 0.5).normalized()                    # [] Line
            yield (moved, second, midline, blend(fixed, moved, parameter).inverse(),
                   blend(fixed, moved, shares).dual().det(), parameter, maximum)

    return fixed, first, shares, moments()


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_animation
    from examples.quadrics.quadric_collision import render

    save_animation(render.animate_collision(*slide(SLIDE)), "quadric_collision", DURATION_MS, forward_then_back=True)

    # checks
    # The poses are apart, touching and overlapping, by the sign of the blend's peak determinant; the
    # contact line is a common tangent, and both ellipses map it to one contact point.
    fixed, moved, _, maximum, contact_line, contact_point = collision()
    assert maximum[0].to_array() > 0 and maximum[2].to_array() < 0
    np.testing.assert_allclose(maximum[1].to_array(), 0, atol=1e-12)
    np.testing.assert_allclose((contact_line & fixed(contact_line)).to_array(), 0, atol=1e-12)
    np.testing.assert_allclose((moved[1](contact_line).normalized() & contact_point).norm().to_array(), 0, atol=1e-12)


if __name__ == "__main__":
    main()
