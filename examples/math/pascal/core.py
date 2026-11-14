"""The quadric through five points and Pascal's theorem in 2D projective geometric algebra.

A quadric is modeled as a polarity map `Plane <- Point` sending each point to its polar plane, and its curve consists of the points that lie on their own polar where `quadric(point) & point == 0`. Because that pairing is linear in the map, each point the quadric passes through imposes one linear condition on it.

Two lines form a degenerate quadric whose pairing vanishes along either line. Four points determine two such line pairs, one from two opposite sides of their quadrilateral and one from its two diagonals, and every quadric through the four is a linear combination of those two pairs. A fifth point imposes one linear condition on the combination and selects a single quadric.

Along any line through the first point, `quadric(point) & point` is quadratic with one root already at the start, so the second crossing is a rational expression in the quadric and the line's direction without a square root. Joining six points on a quadric in order forms a hexagon whose three pairs of opposite sides meet in three collinear crossings (`Pascal's theorem`). Fixing five points and varying the sixth turns the join of those three crossings into a symmetric bilinear form `Scalar <- (Point, Point)` that reproduces the quadric up to scale.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import NumpyContext, stack
from numga.algebras import PGA2D as ga
from numga.topology import Topology

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                                                    # Scalar+
Plane = ga.gatype.vector()                                                     # Vector+
Point = ga.gatype.antivector()                                                 # Antivector-
# Two points join into an intrinsic line, and two lines meet in an extrinsic crossing.
Line = ga.gatype.antibivector()                                                # Antibivector-
Crossing = ga.gatype.bivector()                                                # Bivector+
Quadric = ga.gatype((Plane, Point))                                            # Plane+ <- Point-
Form = ga.gatype((Scalar, Point, Point))                                       # Scalar+ <- (Point-, Point-)

hexagon_loop = Topology.loops([6])


# --- math -----------------------------------------------------------------------------
def quadric(points: Point) -> Quadric:
    """Five points along the trailing axis (`[..., 5] Point-`) determine the unique polarity map (`[...] Plane+ <- Point-`) vanishing on all five by combining the two line-pair quadrics through the first four so that the fifth lies on its own polar."""
    first, second, third, fourth, fifth = (points[..., index] for index in range(5))
    # Two pairs of lines pass through the first four points: opposite sides of their quadrilateral and its diagonals.
    first_lines = stack([first & second, first & third])                       # [2, ...] Line-
    second_lines = stack([third & fourth, second & fourth])                    # [2, ...] Line-
    # Symmetrizing the two lines' pairings builds a polarity map that vanishes along either line.
    line_pairs = (first_lines * (second_lines & Point) + second_lines * (first_lines & Point)) / 2   # [2, ...] Plane+ <- Point-
    side_pair, diagonal_pair = line_pairs[0], line_pairs[1]                    # [...] Plane+ <- Point-, [...] Plane+ <- Point-
    return side_pair * (diagonal_pair(fifth) & fifth) - diagonal_pair * (side_pair(fifth) & fifth)   # [...] Plane+ <- Point-


def hexagon_configurations(five: Point, angles: np.ndarray) -> Iterator[tuple[Point, Line, Crossing, Line]]:
    """Yields per-angle configurations `(hexagon, sides, crossings, pascal_line)` (`[6] Point-`, `[6] Line-`, `[3] Crossing+`, `[] Line-`) as a line through `five[0]` turns through `angles` and traces the sixth point once around `quadric(five)`."""
    shape = quadric(five)                                                      # [] Plane+ <- Point-
    start = five[0]                                                            # [] Point-
    for angle in angles:
        turn = (mv.xy * (-angle / 2)).exp()                                    # [] Rotor+
        # Turning a point at infinity gives the line's ideal direction point.
        heading = turn >> mv.x.dual()                                          # [] Point-
        # Along `start + t * heading`, the quadratic `2 * t * (shape(start) & heading) + t**2 * (shape(heading) & heading)`
        # already vanishes at `t = 0`, giving the second root rationally.
        sixth = (start * (shape(heading) & heading) - heading * (2 * (shape(start) & heading))).normalized()   # [] Point-
        hexagon = stack([five[index] for index in range(5)] + [sixth])         # [6] Point-
        tails, heads = hexagon_loop.incidence[1, 0] * hexagon.field()          # Point-[E], Point-[E]
        sides = (tails & heads).batch()                                        # [6] Line-
        crossings = sides[:3] ^ sides[3:]                                      # [3] Crossing+
        pascal_line = crossings[0] & crossings[2]                              # [] Line-
        yield hexagon, sides, crossings, pascal_line


def pascal_form(five: Point) -> Form:
    """Five fixed corners (`[..., 5] Point-`) define a symmetric bilinear form (`[...] Scalar+ <- (Point-, Point-)`) on the sixth corner from the regressive join of the three opposite-side crossings, which vanishes whenever those crossings are collinear."""
    first, second, third, fourth, fifth = (five[..., index] for index in range(5))
    return (((first & second) ^ (fourth & fifth))
            & ((second & third) ^ (fifth & Point))
            & ((third & fourth) ^ (Point & first)))                            # [...] Scalar+ <- (Point-, Point-)
