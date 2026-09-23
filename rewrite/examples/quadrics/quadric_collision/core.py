"""Collision of two ellipses in PGA2D, read off the blends of their dual quadrics.

A dual ellipse Q maps lines to points: a line L is tangent where L ∨ Q(L) = 0, and Q(L)
is then its point of contact. A motor moves the whole map, motor >> Q(motor << Line).
The blend (1 - λ) Q1 + λ Q2 of two dual ellipses keeps their common tangents,
and its determinant is a cubic in λ that peaks inside (0, 1): above zero the ellipses are
apart and a separating line exists, at zero they touch, below zero they overlap. At
contact the blend at the peak is singular; its null line is the shared tangent, and both
ellipses map it to the same contact point.
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
Quadric = ga.gatype((Point, Line))                  # dual: contact point <= tangent line
Polarity = ga.gatype((Line, Point))                 # primal: polar line <= point

infinity = mv.w                                     # the line at infinity
origin = mv.xy


def ellipse(rx: float, ry: float) -> Quadric:
    """The dual ellipse with semi-axes rx and ry along x and y, centred at the origin.

    Its principal directions are the ideal points yw and wx:
    Q = rx² yw (yw ∨ L) + ry² wx (wx ∨ L) - xy (xy ∨ L).
    """
    return mv.yw * (Line & mv.yw) * rx**2 + mv.wx * (Line & mv.wx) * ry**2 - origin * (Line & origin)


def motor(tx: float, ty: float, angle: float) -> Motor:
    """The rigid motion rotating by angle about the origin, then translating by (tx, ty)."""
    translator = (infinity.wedge(mv.x * tx + mv.y * ty) * -0.5).exp()
    rotor = (origin * (-angle / 2.0)).exp()
    return translator * rotor


# --- math -----------------------------------------------------------------------------
def tangent_line(Q: Quadric, normal: Line) -> Line:
    """The tangent line of Q with the given outward normal direction, whatever the scale or sign of Q.

    The lines with normal n are n - ∞ d. Tangency, L ∨ Q(L) = 0, is quadratic in the offset d:
    (∞ ∨ Q(∞)) d² - 2 (n ∨ Q(∞)) d + n ∨ Q(n) = 0. Its two roots lie either side of the centre's
    offset (n ∨ Q(∞)) / (∞ ∨ Q(∞)); the outward one lies beyond it along n. Rescaling Q rescales
    all three coefficients alike, so neither root moves.
    """
    n = normal.normalized()
    a, b, c = infinity & Q(infinity), n & Q(infinity), n & Q(n)
    return n - infinity * (b / a + ((b * b - a * c) / (a * a)).square_root())


def blend(Q1: Quadric, Q2: Quadric, lam: Scalar) -> Quadric:
    """The blend (1 - λ) Q1 + λ Q2 of two dual quadrics."""
    return Q1 * (1.0 - lam) + Q2 * lam


def cubic_peak(values: Scalar) -> tuple[Scalar, Scalar]:
    """The maximum of a cubic in λ, sampled at λ = 0, 1, 2 and -1 along the first axis.

    The cubic is concave on [0, 1], so its maximum is the root of the derivative
    3 c3 λ² + 2 c2 λ + c1 where the second derivative is negative. Written as
    c1 / (√(c2² - 3 c3 c1) - c2), that root stays finite as c3 vanishes.
    """
    y0, y1, y2, y3 = values
    c3 = (3.0 * y0 - 3.0 * y1 + y2 - y3) / 6.0
    c2 = -y0 + 0.5 * y1 + 0.5 * y3
    c1 = -0.5 * y0 + y1 - y2 / 6.0 - y3 / 3.0
    c0 = y0
    peak = c1 / ((c2 * c2 - 3.0 * c3 * c1).square_root() - c2)
    return peak, ((c3 * peak + c2) * peak + c1) * peak + c0
