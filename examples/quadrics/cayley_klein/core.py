"""Cayley-Klein geometry in PGA2D: the metric is a quadric you choose.

Projective geometry has joins and meets but no distances. Pick one conic, the absolute,
as a polarity map C from points to lines, and every metric notion follows from C and its
inverse Q alone: distances and angles are cross ratios against the absolute, the
perpendiculars to a line l all pass through its pole Q(l), and a circle is the quadric of
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

# Whole-extensor types (GATypes); map types read output <- inputs. The polarity maps a point
# to its polar line, and the pole map a line to its pole:
Scalar = ga.gatype.scalar()
Point = ga.gatype.antivector()
Line = ga.gatype.vector()
Polarity = ga.gatype((Line, Point))       # Line <- Point
Pole = ga.gatype((Point, Line))           # Point <- Line


def point(coords: np.ndarray) -> Point:
    """Finite points at (..., 2) coordinates: the dual of the homogeneous vector."""
    return (mv("x y", coords) + mv.w).dual()


def invariant(pairing: Extensor, a: Extensor, b: Extensor) -> Scalar:
    """The invariant of a pair against the absolute: their pairing over the root of both self-pairings."""
    return pairing(a, b) / (pairing(a, a) * pairing(b, b)).square_root()


# --- math -----------------------------------------------------------------------------
def triangle(C: Polarity, vertices: Point):
    """The sides, angles, side lengths and area of a triangle, all from the absolute."""
    Q: Pole = C.inverse()

    # The invariant of two points is (P1 & C(P2)) / ((P1 & C(P1)) * (P2 & C(P2))).square_root(),
    # and Cayley's distance is its arccosh. The invariant of two lines is the same with Q, and
    # the angle is its arccos. Both stay inside the algebra until the very last step.
    distance_pairing = -Point.regressive(C(mv.rotor() >> Point))
    angle_pairing = Line.regressive(Q(mv.rotor() >> Line))

    # A triangle: its sides are joins of consecutive vertices, batched, and the angle at
    # each vertex is between the two sides leaving it. Gauss-Bonnet gives the area as the
    # angle defect.
    neighbours = vertices[[1, 2, 0]]
    to_next = vertices.regressive(neighbours)
    to_prev = vertices.regressive(vertices[[2, 0, 1]])
    angles = invariant(angle_pairing, to_next, to_prev).clip(-1.0, 1.0).arccos()
    lengths = invariant(distance_pairing, vertices, neighbours).clip(1.0, np.inf).arccosh()
    area = np.pi - angles.sum()
    return to_next, angles, lengths, area


def perpendicular(C: Polarity, side: Line, P: Point):
    """The pole of a line, the perpendicular to it from P, its foot, and P reflected across the line."""
    # Every line perpendicular to l passes through its pole Q(l), so the perpendicular from
    # P is the join P & Q(l) and the foot is its meet with l. Reflection across l is the
    # harmonic homology centred on the pole, and it preserves the distance to the foot.
    pole = C.solve(side)
    normal = P.regressive(pole)
    foot = side.wedge(normal)
    reflected = P - pole * (2.0 * P.regressive(side) / pole.regressive(side))
    return pole, normal, foot, reflected


def circles(C: Polarity, centres: Point, radii: np.ndarray) -> Polarity:
    """Circles of the given radii about the given centres, as quadrics: one per centre and radius."""
    # Fixing the distance to a centre fixes the invariant, and clearing the square root
    # turns that into a quadric in P: the dyad of the centre's polar minus np.cosh(radii) ** 2
    # times the centre's self-invariant times the absolute. One expression, batched over the
    # radii, drawn as the level set P & circle(P) == 0.
    centre = centres[:, None]
    polar = C(centre)
    return polar * polar.regressive(Point) - centre.regressive(polar) * np.cosh(radii) ** 2 * C
