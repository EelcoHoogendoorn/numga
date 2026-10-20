"""Poncelet's porism, in PGA2D: a polygon inscribed in one conic and circumscribed about another
either closes from every starting point or from none.

Two conics, an outer and an inner one. From a point on the outer conic draw a tangent to the inner
one, follow it to where it meets the outer conic again, draw the other tangent from there, and go on.
The porism: if the path closes after some number of sides for one starting point, it closes after
the same number for every starting point.

Points are positions, intrinsic antivectors, and the outer conic is a map from a point to its polar,
`Conic = Plane <- Point`, zero on its points where `conic(point) & point` vanishes. The sides of the
path are chords, joins of two points, intrinsic too. The inner conic is held by the chords that touch
it: it sends a chord to its pole, the meet of the polars of the chord's points, `Envelope = Pole <-
Chord`, and a chord touches it where it passes through its own pole. Five chords fix a conic they
touch, as five points fix a conic through them: two poles make an envelope, the pair of them, zero on
the chords through either; the pairs of opposite corners of four of the chords span every envelope
touching those four, and the fifth picks one.

A chord through a vertex is the vertex joined with a direction, a point at infinity. Along a direction
the outer conic is a quadratic with one root at the vertex, so the other crossing is rational in the
conic. Through a vertex pass two tangents of the inner conic, a quadratic in the direction; knowing one,
the other is rational in the envelope, given any second direction through the vertex that is never a
tangent, such as the direction from a point inside the inner conic. The first tangent from a start, with
neither root known, takes a square root.

In matrix notation the conic and its envelope read as a symmetric matrix and its adjugate. In the
language of elliptic curves, the pairs of a point on the outer conic and a tangent of the inner one
through it form a curve of genus one, a step of the path reads as adding a fixed point on it, and
closing after n sides as that point having order n.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import NumpyContext, stack
from numga.algebras import PGA2D as ga

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()
# Positions are points, intrinsic; polars are lines read by which side a point is on, extrinsic.
Point = ga.gatype.antivector()
Plane = ga.gatype.vector()
# A chord is the join of two points, a line oriented along itself; a pole is the meet of two lines.
Chord = (Point & Point).output_gatype
Pole = (Plane ^ Plane).output_gatype
Conic = ga.gatype((Plane, Point))                                              # Plane <- Point
Envelope = ga.gatype((Pole, Chord))                                            # Pole <- Chord
Motor = ga.gatype.rotor()
# Both roots of a quadratic, one per sign.
SIGNS = np.array([1.0, -1.0])
# The points at infinity along x and along y.
HEADINGS = stack([mv.x.dual(), mv.y.dual()])                                   # [2] Point


# --- math -----------------------------------------------------------------------------
def ellipse(semi: np.ndarray, motor: Motor) -> Conic:
    """The ellipse with the given semi-axes along x and y, carried from the origin by a motor: each
    coordinate line through the origin weighted by the inverse square of its semi-axis, less the line at
    infinity."""
    axes = stack([mv.x, mv.y])                                                 # [2] Plane
    at_origin = (axes * (axes & Point) / semi**2).sum(axis=-1) - mv.w * (mv.w & Point)   # [] Plane <- Point
    return motor >> at_origin(motor << Point)                                  # [] Plane <- Point


def pair(first: Pole, second: Pole) -> Envelope:
    """Two poles as one envelope: zero on the chords through either, its pairing the product of the
    chord's pairings with the two."""
    return (first * (second & Chord) + second * (first & Chord)) / 2           # [...] Pole <- Chord


def envelope(sides: Chord) -> Envelope:
    """The envelope touching five chords on the last axis: the combination of the two pairs of
    opposite corners of the first four that vanishes on the fifth."""
    first, second, third, fourth, fifth = (sides[..., index] for index in range(5))
    corners = pair(first ^ second, third ^ fourth)                             # [...] Pole <- Chord
    diagonals = pair(first ^ third, second ^ fourth)                           # [...] Pole <- Chord
    return corners * (diagonals(fifth) & fifth) - diagonals * (corners(fifth) & fifth)   # [...] Pole <- Chord


def touching(inner: Envelope, vertex: Point, first: Point, second: Point) -> Scalar:
    """The envelope's pairing of the chords through a vertex along two directions: zero for one
    direction alone exactly when its chord touches the inner conic."""
    return (vertex & first) & inner(vertex & second)                           # [...] Scalar


def tangents(inner: Envelope, points: Point) -> Point:
    """The directions of the two tangents of an envelope through each point, on the last axis: the
    roots of its quadratic along the directions, by a square root."""
    along_x, along_y = HEADINGS[0], HEADINGS[1]
    a = touching(inner, points, along_x, along_x)                              # [...] Scalar
    b = touching(inner, points, along_x, along_y)                              # [...] Scalar
    c = touching(inner, points, along_y, along_y)                              # [...] Scalar
    root = discriminant(inner, points).square_root()                           # [...] Scalar
    # The roots of `a + 2 * b * t + c * t * t` in t, as the weights of the two directions.
    return along_x * (root[..., None] * SIGNS - b[..., None]) + along_y * a[..., None]   # [..., 2] Point


def discriminant(inner: Envelope, points: Point) -> Scalar:
    """The discriminant of the envelope's quadratic along the directions through each point: positive
    where two tangents pass, outside the inner conic, and zero on it, where they coincide."""
    along_x, along_y = HEADINGS[0], HEADINGS[1]
    b = touching(inner, points, along_x, along_y)                              # [...] Scalar
    return b * b - touching(inner, points, along_x, along_x) * touching(inner, points, along_y, along_y)   # [...] Scalar


def other_crossing(outer: Conic, vertex: Point, heading: Point) -> Point:
    """Where the chord from a point on the conic along a direction meets the conic again. Along
    `vertex + heading * t` the pairing is `2 * t * (outer(vertex) & heading) + t * t * (outer(heading) & heading)`,
    zero at the vertex and at the other root."""
    return vertex * (outer(heading) & heading) - heading * (2 * (outer(vertex) & heading))   # [...] Point


def other_tangent(inner: Envelope, vertex: Point, heading: Point, turned: Point) -> Point:
    """The direction of the other tangent through a vertex, given the direction of one and a second
    direction that is never a tangent: the twin of `other_crossing`, on directions. Along
    `heading + turned * t` the pairing is `2 * t * touching(heading, turned) + t * t * touching(turned, turned)`,
    zero at the known tangent and at the other root."""
    return (heading * touching(inner, vertex, turned, turned)
            - turned * (2 * touching(inner, vertex, heading, turned)))         # [...] Point


def path(outer: Conic, inner: Envelope, start: Point, heading: Point, inside: Point) -> Iterator[Point]:
    """The vertices of the path on the outer conic, from a start on it along the direction of a tangent
    of the inner conic, without end: where that chord meets the outer conic again, then the other
    tangent through the new vertex, turned against the direction from a point inside the inner conic."""
    vertex = start.broadcast_to(heading.shape)                                 # [...] Point
    while True:
        yield vertex
        vertex = other_crossing(outer, vertex, heading)                        # [...] Point
        vertex = vertex / (mv.w & vertex)                                      # [...] Point
        heading = other_tangent(inner, vertex, heading, vertex - inside)       # [...] Point
        # A direction has no weight; its length is the norm of its complement.
        heading = heading / heading.dual().norm()                              # [...] Point
