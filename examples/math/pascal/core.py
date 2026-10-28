"""The quadric through five points, and Pascal's theorem, in PGA2D.

A quadric is a map from points to planes, `Quadric = Plane <- Point`: the plane it sends a point to is
that point's polar, and the quadric is the set of points that lie on their own polar, where
`quadric(point) & point` vanishes. The pairing is linear in the quadric, so every point it must pass
through is one linear condition on it.

Two lines make a quadric, the pair of them. Through four points pass two such pairs, one pair of
opposite sides of the quadrilateral and the other; every quadric through the four is a combination of
those two, the pencil of quadrics through four points. A fifth point is one more condition on the
combination, and it leaves one quadric.

A sixth point on the quadric follows from any line through the first: the pairing along the line is a
quadratic with one root known, so the other root is rational in the quadric, with no square root.

Pascal's theorem says that for six points on a quadric taken as a hexagon, the three pairs of opposite
sides meet in three points on one line. Read backwards the theorem is a construction: with five points
fixed, the condition that those three crossings lie on one line is quadratic in the sixth point, and
it is the quadric through the five.

In matrix notation a quadric reads as a symmetric three-by-three matrix, the pencil as the combination
of two rank-two matrices, and the three crossings on one line as a vanishing determinant.
"""

from __future__ import annotations

from numga import NumpyContext, stack
from numga.algebras import PGA2D as ga

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                                                    # Scalar+
Plane = ga.gatype.vector()                                                     # Vector+
Point = ga.gatype.antivector()                                                 # Antivector-
# The join of two points is a line, an intrinsic antibivector, and two such lines cross in an extrinsic bivector.
Line = ga.gatype.antibivector()                                                # Antibivector-
Crossing = ga.gatype.bivector()                                                # Bivector+
Quadric = ga.gatype((Plane, Point))                                            # Plane+ <- Point-
Form = ga.gatype((Scalar, Point, Point))                                       # Scalar+ <- (Point-, Point-)


# --- math -----------------------------------------------------------------------------
def quadric(points: Point) -> Quadric:
    """The quadric through five points on the last axis: the combination of the two line pairs through
    the first four that vanishes on the fifth."""
    first, second, third, fourth, fifth = (points[..., index] for index in range(5))
    # Through the first four points pass two pairs of lines: two opposite sides of their quadrilateral,
    # and its two diagonals.
    lines = stack([first & second, first & third])                             # [pairs, ...] Line-
    partners = stack([third & fourth, second & fourth])                        # [pairs, ...] Line-
    # Two lines make a quadric, the pair of them, whose value at a point is the product of the two lines'
    # pairings with it, zero on either line.
    pairs = (lines * (partners & Point) + partners * (lines & Point)) / 2      # [pairs, ...] Plane+ <- Point-
    sides, diagonals = pairs[0], pairs[1]                                      # [...] Plane+ <- Point-, Plane+ <- Point-
    return sides * (diagonals(fifth) & fifth) - diagonals * (sides(fifth) & fifth)   # [...] Plane+ <- Point-


def pascal(five: Point) -> Form:
    """Pascal's condition on a sixth point after five, a form on the sixth point, which appears in two of
    the hexagon's sides: the join of the three crossings of opposite sides, zero exactly when they lie
    on one line."""
    first, second, third, fourth, fifth = (five[..., index] for index in range(5))
    return (((first & second) ^ (fourth & fifth))
            & ((second & third) ^ (fifth & Point))
            & ((third & fourth) ^ (Point & first)))                            # [...] Scalar+ <- (Point-, Point-)
