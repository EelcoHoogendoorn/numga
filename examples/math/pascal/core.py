"""The conic through five points, and Pascal's theorem, in PGA2D.

A conic is a map from points to the hyperplanes of PGA2D, the lines of the plane,
`Conic = Plane <- Point`: the line it sends a point to is that point's polar, and the conic is the
set of points that lie on their own polar, where `conic(point) & point` vanishes. The pairing is
linear in the conic, so every point it must pass through is one linear condition on it.

Two lines make a conic, the pair of them. Through four points pass two such pairs, one pair of
opposite sides of the quadrilateral and the other; every conic through the four is a combination of
those two, the pencil of conics through four points. A fifth point is one more condition on the
combination, and it leaves one conic.

A sixth point on the conic follows from any line through the first: the pairing along the line is a
quadratic with one root known, so the other root is rational in the conic, with no square root.

Pascal's theorem: for six points on a conic taken as a hexagon, the three pairs of opposite sides
meet in three points on one line. Read backwards the theorem is a construction: with five points
fixed, the condition that those three crossings lie on one line is quadratic in the sixth point, and
it is the conic through the five.

In matrix notation a conic reads as a symmetric three-by-three matrix, the pencil as the combination
of two rank-two matrices, and the three crossings on one line as a vanishing determinant.
"""

from __future__ import annotations

import numpy as np

from numga import NumpyContext
from numga.algebras import PGA2D as ga

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()
Plane = ga.gatype.vector()
Point = ga.gatype.antivector()
Conic = ga.gatype((Plane, Point))                                              # Plane <- Point
Form = ga.gatype((Scalar, Point, Point))                                       # Scalar <- (Point, Point)
# The next vertex of each vertex of a hexagon.
NEXT = np.array([1, 2, 3, 4, 5, 0])


# --- math -----------------------------------------------------------------------------
def pair(first: Plane, second: Plane) -> Conic:
    """Two lines as one conic: zero on the points of either line, its pairing the product of the two
    lines' pairings with the point."""
    return (first * (second & Point) + second * (first & Point)) / 2           # [...] Plane <- Point


def conic(points: Point) -> Conic:
    """The conic through five points on the last axis: the combination of the two line pairs through
    the first four that vanishes on the fifth."""
    first, second, third, fourth, fifth = (points[..., index] for index in range(5))
    sides = pair(first & second, third & fourth)                               # [...] Plane <- Point
    diagonals = pair(first & third, second & fourth)                           # [...] Plane <- Point
    return sides * (diagonals(fifth) & fifth) - diagonals * (sides(fifth) & fifth)   # [...] Plane <- Point


def second_crossing(shape: Conic, start: Point, heading: Point) -> Point:
    """Where the line from a point on a conic, towards a point at infinity, meets the conic again. Along
    `start + heading * t` the pairing is `2 * t * (shape(start) & heading) + t * t * (shape(heading) & heading)`,
    zero at the start and at the other root."""
    return start * (shape(heading) & heading) - heading * (2 * (shape(start) & heading))   # [...] Point


def crossings(hexagon: Point) -> Point:
    """The three points where opposite sides of a hexagon meet, on the last axis."""
    sides = hexagon & hexagon[..., NEXT]                                       # [..., 6] Plane
    return sides[..., :3] ^ sides[..., 3:]                                     # [..., 3] Point


def pascal(five: Point) -> Form:
    """Pascal's condition on a sixth point after five, as a form with the sixth point open in both
    places it appears: the join of the three crossings of opposite sides, zero exactly when they lie
    on one line."""
    first, second, third, fourth, fifth = (five[..., index] for index in range(5))
    return (((first & second) ^ (fourth & fifth))
            & ((second & third) ^ (fifth & Point))
            & ((third & fourth) ^ (Point & first)))                            # [...] Scalar <- (Point, Point)


# --- plumbing -------------------------------------------------------------------------
def point(coords: np.ndarray) -> Point:
    """Points at unit weight from their x and y coordinates."""
    return (mv("x y", coords) + mv.w).dual()


def headings(angles: np.ndarray) -> Point:
    """Points at infinity in the directions at the given angles."""
    return mv("x y", np.stack([np.cos(angles), np.sin(angles)], axis=-1)).dual()
