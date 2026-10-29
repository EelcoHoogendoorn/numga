"""Poncelet's porism, in PGA2D: a polygon inscribed in one quadric and circumscribed about another
either closes from every starting point or from none.

The path runs between two quadrics, an outer and an inner one. From a point on the outer quadric it follows
a tangent of the inner one to where it meets the outer quadric again, takes the other tangent from there,
and goes on. Poncelet's porism states that if the path closes after some number of sides for one
starting point, it closes after the same number for every starting point.

Points are positions, intrinsic antivectors, and the outer quadric is a map from a point to its polar,
`Quadric = Plane <- Point`, zero on its points where `quadric(point) & point` vanishes. The sides of the
path are chords, joins of two points, intrinsic too. The inner quadric is held by the chords that touch
it: it sends a chord to its pole, the meet of the polars of the chord's points,
`Envelope = Pole <- Chord`, and a chord touches it where it passes through its own pole. Five chords
fix a quadric they touch, as five points fix a quadric through them: two poles make an envelope, the pair
of them, zero on the chords through either; the pairs of opposite corners of four of the chords span
every envelope touching those four, and the fifth picks one.

A chord through a vertex is the vertex joined with a direction, a point at infinity. Along a direction
the outer quadric is a quadratic with one root at the vertex, so the other crossing is rational in the
quadric. Through a vertex pass two tangents of the inner quadric, a quadratic in the direction; knowing one,
the other is rational in the envelope, given any second direction through the vertex that is never a
tangent, such as the direction from a point inside the inner quadric. The first tangent from a start, with
neither root known, takes a square root.

In matrix notation the quadric and its envelope read as a symmetric matrix and its adjugate. In the
language of elliptic curves, the pairs of a point on the outer quadric and a tangent of the inner one
through it form a curve of genus one, a step of the path reads as adding a fixed point on it, and
closing after n sides as that point having order n.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import NumpyContext, stack
from numga.algebras import PGA2D as ga

mv = NumpyContext(ga).multivector
# Positions are points, intrinsic; polars are planes, read by which side a point is on, extrinsic.
Point = ga.gatype.antivector()                                                 # Antivector-
Plane = ga.gatype.vector()                                                     # Vector+
# A chord is the join of two points, a line oriented along itself; a pole is the meet of two planes.
Chord = (Point & Point).output_gatype                                          # Antibivector-
Pole = (Plane ^ Plane).output_gatype                                           # Bivector+
Quadric = ga.gatype((Plane, Point))                                            # Plane+ <- Point-
Envelope = ga.gatype((Pole, Chord))                                            # Pole+ <- Chord-
Motor = ga.gatype.rotor()                                                      # Rotor+
# Both roots of a quadratic, one per sign.
signs = np.array([1, -1])


# --- math -----------------------------------------------------------------------------
def ellipsoid(semi: np.ndarray, motor: Motor) -> Quadric:
    """The ellipsoid with the given semi-axes along x and y, carried from the origin by a motor: each
    coordinate plane through the origin weighted by the inverse square of its semi-axis, less the plane at
    infinity."""
    axes = stack([mv.x, mv.y])                                                 # [2] Plane+
    at_origin = (axes * (axes & Point) / semi**2).sum(axis=-1) - mv.w * (mv.w & Point)   # [] Plane+ <- Point-
    return motor >> at_origin(motor << Point)                                  # [] Plane+ <- Point-


def envelope(sides: Chord) -> Envelope:
    """The envelope touching five chords on the last axis: the combination of the two pairs of
    opposite corners of the first four that vanishes on the fifth."""
    first, second, third, fourth, fifth = (sides[..., index] for index in range(5))
    # Two opposite corners of the quadrilateral of the first four chords, and two others.
    p, q = first ^ second, third ^ fourth                                      # [...] Pole+, Pole+
    r, s = first ^ third, second ^ fourth                                      # [...] Pole+, Pole+
    # Two poles as one envelope, zero on the chords through either: the product of a chord's pairings
    # with the two, symmetric in them.
    corners = (p * (q & Chord) + q * (p & Chord)) / 2                          # [...] Pole+ <- Chord-
    diagonals = (r * (s & Chord) + s * (r & Chord)) / 2                        # [...] Pole+ <- Chord-
    return corners * (diagonals(fifth) & fifth) - diagonals * (corners(fifth) & fifth)   # [...] Pole+ <- Chord-


def tangents(inner: Envelope, points: Point) -> Point:
    """The directions of the two tangents of an envelope through each point, on the last axis: the
    roots of its quadratic along the directions, by a square root."""
    # The points at infinity along x and along y, and the chords through each point towards them.
    x_direction, y_direction = mv.x.dual(), mv.y.dual()                        # [] Point-, Point-
    x_chords, y_chords = points & x_direction, points & y_direction            # [...] Chord-, Chord-
    a = x_chords & inner(x_chords)                                             # [...] Scalar+
    b = x_chords & inner(y_chords)                                             # [...] Scalar+
    c = y_chords & inner(y_chords)                                             # [...] Scalar+
    # The discriminant is positive where two tangents pass, outside the inner quadric, and zero on it,
    # where they coincide.
    root = (b * b - a * c).square_root()                                       # [...] Scalar+
    # The roots of `a + 2 * b * t + c * t * t` in t, as the weights of the two directions.
    return x_direction * (root[..., None] * signs - b[..., None]) + y_direction * a[..., None]   # [..., 2] Point-


def path(outer: Quadric, inner: Envelope, start: Point, heading: Point, inside: Point) -> Iterator[Point]:
    """The vertices of the path on the outer quadric, from a start on it along the direction of a tangent
    of the inner quadric, without end: where that chord meets the outer quadric again, then the other
    tangent through the new vertex, turned against the direction from a point inside the inner quadric."""
    vertex = start.broadcast_to(heading.shape)                                 # [...] Point-
    while True:
        yield vertex
        # Along `vertex + heading * t` the outer quadric's pairing is
        # `2 * t * (outer(vertex) & heading) + t * t * (outer(heading) & heading)`, zero at the vertex and at the other root.
        crossing = vertex * (outer(heading) & heading) - heading * (2 * (outer(vertex) & heading))   # [...] Point-
        vertex = crossing.normalized()                                         # [...] Point-
        # The incoming side, and the chord from a point inside the inner quadric, which is never a tangent.
        spoke = vertex - inside                                                # [...] Point-
        side, spoke_line = vertex & heading, vertex & spoke                    # [...] Chord-, Chord-
        # Along `heading + spoke * t` the envelope's pairing of the chords through the vertex is
        # `2 * t * (side & inner(spoke_line)) + t * t * (spoke_line & inner(spoke_line))`, zero at the incoming side and at the other tangent.
        tangent = heading * (spoke_line & inner(spoke_line)) - spoke * (2 * (side & inner(spoke_line)))   # [...] Point-
        # A direction has no weight; its length is the norm of its complement.
        heading = tangent / tangent.dual().norm()                              # [...] Point-
