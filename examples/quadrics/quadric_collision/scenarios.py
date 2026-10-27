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
    first_shape = ellipse(*FIRST_SEMI)                                         # [] Point- <- Line+
    first_ellipse = placement >> first_shape(placement << Line)                # [] Point- <- Line+

    # The second ellipse, turned, is translated so that its tangent point with the opposite normal lands
    # on the first ellipse's tangent point with the chosen normal.
    normal = (mv.xy * (-CONTACT_NORMAL / 2)).exp() >> mv.x                     # [] Line+
    second_ellipse = ellipse(*SECOND_SEMI)                                     # [] Point- <- Line+
    turn = motor(0.0, 0.0, SECOND_TURN)
    turned_second = turn >> second_ellipse(turn << Line)                       # [] Point- <- Line+
    target = first_ellipse(tangent_line(first_ellipse, normal))                # [] Point-
    start = turned_second(tangent_line(turned_second, -normal))                # [] Point-
    # Their displacement is the difference of the two points taken at unit weight.
    displacement = target.normalized() - start.normalized()                    # [] Point-
    translator = ((infinity ^ displacement.dual()) * -0.5).exp()               # [] Motor+
    touch = translator * turn                                                  # [] Motor+
    return first_ellipse, second_ellipse, normal, touch


def collision():
    """A second ellipse touching the first along a chosen normal, then moved apart and into it."""
    first_ellipse, second_ellipse, normal, touch = pair()
    # The poses along the normal: apart, touching, and into the first.
    shifts = ((infinity ^ (normal * OFFSETS)) * -0.5).exp()                    # [poses] Motor+
    poses = shifts * touch                                                     # [poses] Motor+
    placed_second = poses >> second_ellipse(poses << Line)                     # [poses] Point- <- Line+

    # Four samples of the blend's determinant fix the cubic exactly; only locating its peak leaves the
    # algebra.
    parameter, maximum = cubic_peak(blend(first_ellipse, placed_second, mv.scalar(SAMPLES)[:, None]).dual().det())   # [poses] Scalar+, Scalar+

    # At contact the blend at the peak has a null line, its last singular vector: the shared tangent, which
    # each ellipse maps to the same contact point.
    _, _, lines = blend(first_ellipse, placed_second[1], parameter[1]).dual().svd()
    contact_line = lines[-1].normalized()                                      # [] Line+
    contact_point = first_ellipse(contact_line).normalized()                   # [] Point-
    return first_ellipse, placed_second, parameter, maximum, contact_line, contact_point


def slide(offsets: np.ndarray) -> tuple[Quadric, Line, Scalar, Iterator[tuple]]:
    """The first ellipse, its tangent line with the chosen normal, the shares the determinant is drawn at, and the
    second ellipse at each offset along the normal from touching, with what the figure draws of it."""
    first_ellipse, second_ellipse, normal, touch = pair()
    first_tangent = tangent_line(first_ellipse, normal)                        # [] Line+
    shares = mv.scalar(np.linspace(0.001, 0.999, N_SHARES)[:, None])           # [shares] Scalar+

    def moments() -> Iterator[tuple[Quadric, Line, Line, Polarity, Scalar, Scalar, Scalar]]:
        # Each moment: the second ellipse, its tangent line with the opposite normal and the midline of the two
        # tangents, the blend at the determinant's peak as a map from points to their polar lines, and the
        # determinant along the blends with its peak.
        for offset in offsets:
            shift = ((infinity ^ (normal * offset)) * -0.5).exp()              # [] Motor+
            pose = shift * touch                                               # [] Motor+
            placed_second = pose >> second_ellipse(pose << Line)               # [] Point- <- Line+
            parameter, maximum = cubic_peak(blend(first_ellipse, placed_second, mv.scalar(SAMPLES)).dual().det())   # [] Scalar+, Scalar+
            second_tangent = tangent_line(placed_second, -normal)              # [] Line+
            midline = ((first_tangent - second_tangent) * 0.5).normalized()    # [] Line+
            yield (placed_second, second_tangent, midline, blend(first_ellipse, placed_second, parameter).inverse(),
                   blend(first_ellipse, placed_second, shares).dual().det(), parameter, maximum)

    return first_ellipse, first_tangent, shares, moments()


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_animation
    from examples.quadrics.quadric_collision import render

    save_animation(render.animate_collision(*slide(SLIDE)), "quadric_collision", DURATION_MS, forward_then_back=True)

    # checks
    # The poses are apart, touching and overlapping, by the sign of the blend's peak determinant; the
    # contact line is a common tangent, and both ellipses map it to one contact point.
    first_ellipse, placed_second, _, maximum, contact_line, contact_point = collision()
    assert maximum[0] > 0 and maximum[2] < 0
    np.testing.assert_allclose(maximum[1].to_array(), 0, atol=1e-12)
    np.testing.assert_allclose((contact_line & first_ellipse(contact_line)).to_array(), 0, atol=1e-12)
    np.testing.assert_allclose((placed_second[1](contact_line).normalized() & contact_point).norm().to_array(), 0, atol=1e-12)


if __name__ == "__main__":
    main()
