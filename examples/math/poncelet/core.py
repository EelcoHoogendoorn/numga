"""Poncelet's porism, in PGA2D: a polygon inscribed in one conic and circumscribed about another
either closes from every starting point or from none.

Two conics, an outer and an inner one. From a point on the outer conic draw a tangent to the inner
one, follow it to where it meets the outer conic again, draw the other tangent from there, and go on.
The porism: if the path closes after some number of sides for one starting point, it closes after
the same number for every starting point.

The outer conic is a map from points to lines, `Conic = Plane <- Point`, zero on its points where
`conic(point) & point` vanishes. The inner conic is held by its tangent lines, a map the other way,
`Envelope = Point <- Plane`, zero on the lines that touch it. Everything about one has a twin about
the other with points and lines exchanged, and a step of the path is one such pair of twins. A
line meets the outer conic twice; knowing one crossing, the other is rational in the conic. Through
a point pass two tangents of the inner conic; knowing one, the other is rational in the envelope.
The same formula does both, once on points and once on lines.

Five lines fix a conic they touch, as five points fix a conic through them: two points make an
envelope, the pair of them, zero on the lines through either; the pairs of opposite corners of four
of the lines span every envelope touching those four, and the fifth line picks one. With the five
sides of a pentagon inscribed in the outer conic, the inner conic closes one pentagon, and by the
porism every other one started on the outer conic.

In matrix notation the conic and its envelope read as a symmetric matrix and its adjugate. In the
language of elliptic curves, the pairs of a point on the outer conic and a tangent of the inner one
through it form a curve of genus one, a step of the path reads as adding a fixed point on it, and
closing after n sides as that point having order n.
"""

from __future__ import annotations

import numpy as np

from numga import NumpyContext, stack
from numga.algebras import PGA2D as ga

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()
Plane = ga.gatype.vector()
Point = ga.gatype.antivector()
Conic = ga.gatype((Plane, Point))                                              # Plane <- Point
Envelope = ga.gatype((Point, Plane))                                           # Point <- Plane
# Both roots of a quadratic, one per sign.
SIGNS = np.array([1.0, -1.0])
# The points at infinity along x and along y.
HEADINGS = mv("x y", np.eye(2)).dual()                                         # [2] Point


# --- math -----------------------------------------------------------------------------
def pair(first: Point, second: Point) -> Envelope:
    """Two points as one envelope: zero on the lines through either, its pairing the product of the
    line's pairings with the two points."""
    return (first * (second & Plane) + second * (first & Plane)) / 2          # [...] Point <- Plane


def envelope(sides: Plane) -> Envelope:
    """The envelope touching five lines on the last axis: the combination of the two pairs of
    opposite corners of the first four that vanishes on the fifth."""
    first, second, third, fourth, fifth = (sides[..., index] for index in range(5))
    corners = pair(first ^ second, third ^ fourth)                             # [...] Point <- Plane
    diagonals = pair(first ^ third, second ^ fourth)                           # [...] Point <- Plane
    return corners * (diagonals(fifth) & fifth) - diagonals * (corners(fifth) & fifth)   # [...] Point <- Plane


def roots(a: Scalar, b: Scalar, c: Scalar) -> Scalar:
    """The two roots of the quadratic `a + 2 * b * t + c * t * t` along the combinations of two ends,
    as the weights of the ends on the last axis, one root per sign on the axis before it:
    `first * (sign * root - b) + second * a`, with `root * root == b * b - a * c`."""
    root = (b * b - a * c).square_root()                                       # [...] Scalar
    first = root[..., None] * SIGNS - b[..., None]                             # [..., 2] Scalar
    return stack([first, a[..., None].broadcast_to(first.shape)], axis=-1)     # [..., 2, 2] Scalar


def tangents(inner: Envelope, points: Point) -> Plane:
    """The two tangents of an envelope through each point, on the last axis: its zeros along the lines
    through the point."""
    ends = points[..., None] & HEADINGS                                        # [..., 2] Plane
    first, second = ends[..., 0], ends[..., 1]                                 # [...] Plane
    weights = roots(inner(first) & first, inner(first) & second, inner(second) & second)   # [..., 2, 2] Scalar
    return (weights * ends[..., None, :]).sum(axis=-1).normalized()            # [..., 2] Plane


def other_crossing(outer: Conic, vertex: Point, heading: Point) -> Point:
    """Where the line from a point on the conic towards a heading meets the conic again. Along
    `vertex + heading * t` the pairing is `2 * t * (outer(vertex) & heading) + t * t * (outer(heading) & heading)`,
    zero at the vertex and at the other root."""
    return vertex * (outer(heading) & heading) - heading * (2 * (outer(vertex) & heading))   # [...] Point


def other_tangent(inner: Envelope, side: Plane, turned: Plane) -> Plane:
    """The other tangent through the point where a tangent meets another line: the twin of
    `other_crossing`, with lines for points. Along `side + turned * t` the pairing is
    `2 * t * (inner(side) & turned) + t * t * (inner(turned) & turned)`, zero at the side and at the
    other root."""
    return side * (inner(turned) & turned) - turned * (2 * (inner(side) & turned))   # [...] Plane


def path(outer: Conic, inner: Envelope, start: Point, side: Plane, steps: int) -> Point:
    """The vertices of the path on the outer conic, from a start on it along a tangent of the inner
    conic through it, on the first axis: at each vertex the other tangent through it, and where that
    meets the outer conic again."""
    vertices = [start.broadcast_to(side.shape)]
    for _ in range(steps):
        side = other_tangent(inner, side, side | vertices[-1]).normalized()    # [...] Plane
        vertex = other_crossing(outer, vertices[-1], side ^ mv.w)              # [...] Point
        vertices.append(vertex / (mv.w & vertex))
    return stack(vertices)                                                     # [steps + 1, ...] Point


# --- plumbing -------------------------------------------------------------------------
def point(coords: np.ndarray) -> Point:
    """Points at unit weight from their x and y coordinates."""
    return (mv("x y", coords) + mv.w).dual()


def ellipse(centre: np.ndarray, semi: np.ndarray, tilt: float) -> Conic:
    """The ellipse about a centre with the given semi-axes, its first axis turned from x by the tilt:
    the lines through the centre along each axis, weighted by the inverse square of the semi-axis, less
    the line at infinity."""
    turn = np.array([[np.cos(tilt), np.sin(tilt)], [-np.sin(tilt), np.cos(tilt)]])
    axes = mv("x y", turn)                                                     # [2] Plane
    through = axes - mv.w * (axes & point(centre))                             # [2] Plane
    return (through * (through & Point) / semi**2).sum(axis=-1) - mv.w * (mv.w & Point)
