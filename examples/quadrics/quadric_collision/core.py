"""Collision of two ellipsoids in PGA2D is decided from the blends of their dual quadrics.

A dual ellipsoid maps planes to points: a plane is tangent where `plane & dual(plane) == 0`, and `dual(plane)` is its point of contact. A motor moves the whole map as `motor >> dual(motor << Plane)`. A blend of two dual ellipsoids, `first * (1 - share) + second * share`, preserves their common tangents, and its determinant is a cubic in the share that peaks between 0 and 1: above zero the ellipsoids stand apart with a separating plane between them, at zero they touch, and below zero they overlap. At contact the blend at the peak is singular; its null plane is the shared tangent, and both ellipsoids map it to the same contact point.
"""

from __future__ import annotations

from numga import NumpyContext
from numga.algebras import PGA2D

ga = PGA2D
ctx = NumpyContext(ga)
mv = ctx.multivector

Scalar = ga.gatype.scalar()                                                    # Scalar+
Plane = ga.gatype.vector()                                                     # Vector+
Point = ga.gatype.antivector()                                                 # Antivector-
Motor = ga.gatype.rotor()                                                      # Rotor+
# The dual quadric maps a tangent plane to its contact point; the primal polarity maps a point
# to its polar plane.
Quadric = ga.gatype((Point, Plane))                                            # Point- <- Plane+
Polarity = ga.gatype((Plane, Point))                                           # Plane+ <- Point-

# The plane at infinity, the unit-weight origin, and the coordinate directions along x and y.
infinity = mv.w                                                                # [] Plane+
origin = mv.w.dual()                                                           # [] Point-
direction_x, direction_y = mv.direction([1, 0]), mv.direction([0, 1])          # [] Point-, Point-


# --- math -----------------------------------------------------------------------------
def rest_ellipsoid(semi_x: float, semi_y: float) -> Quadric:
    """An origin-centred dual ellipsoid (`[] Point- <- Plane+`) combines the two axis direction dyads weighted by their squared semi-axes, minus the origin dyad."""
    return direction_x * (Plane & direction_x) * semi_x**2 + direction_y * (Plane & direction_y) * semi_y**2 - origin * (Plane & origin)


def placement_motor(shift_x: float, shift_y: float, angle: float) -> Motor:
    """A rigid motion (`[] Motor+`) turns about the origin by `angle` and translates by `(shift_x, shift_y)`."""
    translator = ((infinity ^ (mv.x * shift_x + mv.y * shift_y)) * -0.5).exp()   # [] Motor+
    rotor = (mv.xy * (-angle / 2)).exp()                                         # [] Motor+
    return translator * rotor


def tangent_plane(dual: Quadric, normal: Plane) -> Plane:
    """The outward tangent plane (`[...] Plane+`) of `dual` along `normal` is invariant under rescaling or negating `dual`.

    Planes sharing `normal`'s direction have the form `unit - infinity * offset`. Tangency is quadratic in `offset`, `a * offset**2 - 2 * b * offset + c == 0`, whose coefficients pair `infinity` and `unit` across `dual`; the two roots straddle the centre offset `b / a`, and the outward root lies beyond it along `unit`.
    """
    unit = normal.normalized()                                                 # [...] Plane+
    a, b, c = infinity & dual(infinity), unit & dual(infinity), unit & dual(unit)   # [...] Scalar+, Scalar+, Scalar+
    return unit - infinity * (b / a + ((b * b - a * c) / (a * a)).square_root())


def quadric_blend(first: Quadric, second: Quadric, share: Scalar) -> Quadric:
    """Linearly interpolating `first` and `second` by `share` forms their intermediate dual quadric (`[...] Point- <- Plane+`)."""
    return first * (1 - share) + second * share


def cubic_peak(values: Scalar) -> tuple[Scalar, Scalar]:
    """Four determinant samples at shares `0, 1, 2, -1` along axis `0` (`[4, ...] Scalar+`) fix the concave cubic between `0` and `1` and locate its peak share and value (`[...] Scalar+, Scalar+`).

    Writing the stationary root of `3 * c3 * share**2 + 2 * c2 * share + c1` as `c1 / ((c2 * c2 - 3 * c3 * c1).square_root() - c2)` keeps the peak finite when the cubic term `c3` vanishes.
    """
    y0, y1, y2, y3 = values                                                    # [...] Scalar+, Scalar+, Scalar+, Scalar+
    c3 = (3 * y0 - 3 * y1 + y2 - y3) / 6                                       # [...] Scalar+
    c2 = -y0 + (y1 + y3) / 2                                                   # [...] Scalar+
    c1 = -y0 / 2 + y1 - y2 / 6 - y3 / 3                                        # [...] Scalar+
    c0 = y0                                                                    # [...] Scalar+
    peak = c1 / ((c2 * c2 - 3 * c3 * c1).square_root() - c2)                   # [...] Scalar+
    return peak, ((c3 * peak + c2) * peak + c1) * peak + c0
