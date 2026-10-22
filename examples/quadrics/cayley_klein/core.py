"""Cayley-Klein geometry in PGA2D: the metric is a quadric you choose.

Projective geometry has joins and meets but no distances. Pick one conic, the absolute,
as a polarity map from points to lines, and every metric notion follows from it and its
inverse alone: distances and angles are cross ratios against the absolute, the
perpendiculars to a line all pass through its pole, and a circle is the quadric of
points at fixed cross ratio from its centre. With the unit circle as the absolute this is
the hyperbolic Beltrami-Klein disk. Flip one sign and the same lines do elliptic geometry.
"""

from __future__ import annotations

import numpy as np

from numga import Extensor, NumpyContext
from numga.algebras import PGA2D

ga = PGA2D
ctx = NumpyContext(ga)
mv = ctx.multivector

# Whole-extensor types (GATypes); map types read output <- inputs. Points are antivectors with intrinsic
# orientation and lines vectors with extrinsic orientation, each with a side, as polars and mirrors are. A
# join of two points is a line oriented along itself instead, intrinsic; times an intrinsic scalar from
# the data, it becomes a line with a side. The polarity maps a point to its polar line, and the pole map
# a line to its pole:
Scalar = ga.gatype.scalar()
Point = ga.gatype.antivector()
Line = ga.gatype.vector()
Polarity = ga.gatype((Line, Point))       # Line <- Point
Pole = ga.gatype((Point, Line))           # Point <- Line


def invariant(pairing: Extensor, a: Extensor, b: Extensor) -> Scalar:
    """The invariant of a pair against the absolute: their pairing over the root of both self-pairings."""
    return pairing(a, b) / (pairing(a, a) * pairing(b, b)).square_root()


# --- math -----------------------------------------------------------------------------
def triangle(absolute: Polarity, vertices: Point):
    """The sides, angles, side lengths and area of a triangle, all from the absolute."""
    dual_absolute = absolute.inverse()                                          # Pole

    # The invariant of two points is (a & absolute(b)) / ((a & absolute(a)) * (b & absolute(b))).square_root(),
    # and Cayley's distance is its arccosh. The invariant of two lines is the same with the dual absolute,
    # and the angle is its arccos. Both stay inside the algebra until the very last step.
    distance_pairing = -(Point & absolute(Point))                         # [] Scalar <- (Point, Point)
    angle_pairing = Line & dual_absolute(Line)                            # [] Scalar <- (Line, Line)

    # A triangle: its sides are joins of consecutive vertices, batched, and the angle at
    # each vertex is between the two sides leaving it. Gauss-Bonnet gives the area as the
    # angle defect.
    neighbours = vertices[[1, 2, 0]]
    # The triangle's winding, an intrinsic scalar: the joins leaving each vertex times it are lines with a
    # side, which the absolute reads, and one factor on both lines of an angle leaves the angle as it is.
    winding = vertices[0] & vertices[1] & vertices[2]                      # [] Scalar
    to_next = (vertices & neighbours) * winding                            # [3] Line
    to_prev = (vertices & vertices[[2, 0, 1]]) * winding                   # [3] Line
    angles = invariant(angle_pairing, to_next, to_prev).clip(-1.0, 1.0).arccos()
    lengths = invariant(distance_pairing, vertices, neighbours).clip(1.0, np.inf).arccosh()
    area = np.pi - angles.sum()
    return to_next, angles, lengths, area


def perpendicular(absolute: Polarity, side: Line, point: Point):
    """The pole of a line, the perpendicular to it from a point, its foot, and the point reflected across
    the line."""
    # Every line perpendicular to a side passes through its pole, so the perpendicular from the point is
    # their join and the foot is its meet with the side. Reflection across the side is the harmonic
    # homology centred on the pole, and it preserves the distance to the foot.
    pole = absolute.solve(side)                                            # [] Point
    normal = point & pole                                                  # [] Line
    foot = side ^ normal                                                   # [] Point
    reflected = point - pole * (2.0 * (point & side) / (pole & side))      # [] Point
    return pole, normal, foot, reflected


def circles(absolute: Polarity, centres: Point, radii: np.ndarray) -> Polarity:
    """Circles of the given radii about the given centres, as quadrics: one per centre and radius."""
    # Fixing the distance to a centre fixes the invariant, and clearing the square root
    # turns that into a quadric in P: the dyad of the centre's polar minus np.cosh(radii) ** 2
    # times the centre's self-invariant times the absolute. One expression, batched over the
    # radii, drawn as the level set P & circle(P) == 0.
    centre = centres[:, None]
    polar = absolute(centre)                                               # [centres, 1] Line
    return polar * (polar & Point) - (centre & polar) * np.cosh(radii) ** 2 * absolute   # [centres, radii] Line <- Point

