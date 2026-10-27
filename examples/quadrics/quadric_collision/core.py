"""Collision of two ellipses in PGA2D, read off the blends of their dual quadrics.

A dual ellipse maps lines to points: a line is tangent where `line & dual(line) == 0`, and
`dual(line)` is then its point of contact. A motor moves the whole map, `motor >> dual(motor << Line)`.
A blend of two dual ellipses, `first * (1 - share) + second * share`, keeps their common tangents, and its
determinant is a cubic in the share that peaks between 0 and 1: above zero the ellipses are apart and a
separating line exists, at zero they touch, below zero they overlap. At contact the blend at the peak is
singular; its null line is the shared tangent, and both ellipses map it to the same contact point.
"""

from __future__ import annotations

from numga import NumpyContext
from numga.algebras import PGA2D

ga = PGA2D
ctx = NumpyContext(ga)
mv = ctx.multivector

Scalar = ga.gatype.scalar()
Line = ga.gatype.vector()
Point = ga.gatype.antivector()
Motor = ga.gatype.rotor()
# The dual quadric maps a tangent line to its contact point; the primal one maps a point to
# its polar line.
Quadric = ga.gatype((Point, Line))                  # Point- <- Line+
Polarity = ga.gatype((Line, Point))                 # Line+ <- Point-

# The line at infinity, the origin, and the directions along x and y.
infinity = mv.w                                     # [] Line+
origin = mv.point([0.0, 0.0])                       # [] Point-
along_x, along_y = mv.direction([1.0, 0.0]), mv.direction([0.0, 1.0])   # [] Point-, Point-


def ellipse(semi_x: float, semi_y: float) -> Quadric:
    """The dual ellipse with the given semi-axes along x and y, centred at the origin: each axis
    direction paired with itself and weighted by its squared semi-axis, less the origin paired with
    itself."""
    return along_x * (Line & along_x) * semi_x**2 + along_y * (Line & along_y) * semi_y**2 - origin * (Line & origin)


def motor(shift_x: float, shift_y: float, angle: float) -> Motor:
    """The rigid motion turning by an angle about the origin, then moving it by the given shifts."""
    translator = ((infinity ^ (mv.x * shift_x + mv.y * shift_y)) * -0.5).exp()
    rotor = (mv.xy * (-angle / 2.0)).exp()
    return translator * rotor


# --- math -----------------------------------------------------------------------------
def tangent_line(dual: Quadric, normal: Line) -> Line:
    """The tangent line of a dual ellipse with the given outward normal, whatever its scale or sign.

    The lines with the normal's direction are `normal - infinity * offset`. Tangency is quadratic in the
    offset: `a * offset**2 - 2 * b * offset + c == 0`, with `a`, `b` and `c` the pairings of the line at
    infinity and the normal through the dual. Its two roots lie either side of the centre's offset
    `b / a`; the outward one lies beyond it along the normal. Rescaling the dual rescales all three
    coefficients alike, so neither root moves.
    """
    unit = normal.normalized()                                                 # [...] Line+
    a, b, c = infinity & dual(infinity), unit & dual(infinity), unit & dual(unit)   # [...] Scalar+, Scalar+, Scalar+
    return unit - infinity * (b / a + ((b * b - a * c) / (a * a)).square_root())


def blend(first: Quadric, second: Quadric, share: Scalar) -> Quadric:
    """The blend of two dual quadrics, the given share of the way from the first to the second."""
    return first * (1.0 - share) + second * share


def cubic_peak(values: Scalar) -> tuple[Scalar, Scalar]:
    """The maximum of a cubic in the share, sampled at 0, 1, 2 and -1 along the first axis.

    The cubic is concave between 0 and 1, so its maximum is the root of the derivative
    `3 * c3 * share**2 + 2 * c2 * share + c1` where the second derivative is negative. Written as
    `c1 / ((c2 * c2 - 3 * c3 * c1).square_root() - c2)`, that root stays finite as c3 vanishes.
    """
    y0, y1, y2, y3 = values
    c3 = (3.0 * y0 - 3.0 * y1 + y2 - y3) / 6.0
    c2 = -y0 + 0.5 * y1 + 0.5 * y3
    c1 = -0.5 * y0 + y1 - y2 / 6.0 - y3 / 3.0
    c0 = y0
    peak = c1 / ((c2 * c2 - 3.0 * c3 * c1).square_root() - c2)
    return peak, ((c3 * peak + c2) * peak + c1) * peak + c0
