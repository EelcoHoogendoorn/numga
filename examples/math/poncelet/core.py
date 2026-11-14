"""Poncelet's porism for polygons inscribed in an outer point quadric and circumscribed about an
inner line envelope.

Starting from a vertex on the outer quadric, a path steps along a tangent of the inner envelope to
its second intersection with the outer quadric, switches to the other tangent through that vertex,
and repeats. Poncelet's porism states that if the path closes after a given number of sides from one
starting point, it closes after the same number of sides from every starting point on the outer
quadric.

Points are modelled as intrinsic antivectors, lines as point joins, polar planes as extrinsic
vectors, and poles as plane meets. An outer point quadric maps each point to its polar plane and
vanishes where a point lies on its own polar, while an inner envelope maps each line to its pole and
vanishes on tangent lines that pass through their own poles. Five tangent lines fix the unique
envelope touching them as a linear combination of two degenerate pole-pair envelopes from the
quadrilateral of the first four lines. Once one crossing on a line or one tangent in a vertex pencil
is known, the second root follows rationally from symmetric bilinear pairings without further square
roots.
"""

from __future__ import annotations

from collections.abc import Iterator
from itertools import islice

import numpy as np

from numga import NumpyContext, stack
from numga.algebras import PGA2D as ga

mv = NumpyContext(ga).multivector
# Points are intrinsic positions, while polar planes are extrinsic linear forms on points.
Point = ga.gatype.antivector()                                                 # Antivector-
Plane = ga.gatype.vector()                                                     # Vector+
# Joining two points forms an intrinsic line, while intersecting two planes forms an extrinsic pole.
Line = ga.gatype(Point & Point)                                                # Antibivector-
Pole = ga.gatype(Plane ^ Plane)                                                # Bivector+
Quadric = ga.gatype((Plane, Point))                                            # Plane+ <- Point-
Envelope = ga.gatype((Pole, Line))                                             # Pole+ <- Line-
Motor = ga.gatype.rotor()                                                      # Rotor+
# The two sign choices select the two quadratic roots in `tangents`.
root_signs = np.array([1, -1])                                                 # [2]


# --- math -----------------------------------------------------------------------------
def ellipsoid(semi_axes: np.ndarray, motor: Motor) -> Quadric:
    """Principal planes carried from the origin by `motor` and weighted by `semi_axes` (`[2]`)
    define the ellipsoid's polarity map (`[] Plane+ <- Point-`)."""
    axes = motor >> stack([mv.x, mv.y])                                        # [2] Plane+
    axial_part = (axes * (axes & Point) / semi_axes**2).sum(axis=-1)           # [] Plane+ <- Point-
    return axial_part - mv.w * (mv.w & Point)                                  # [] Plane+ <- Point-


def envelope(sides: Line) -> Envelope:
    """Five tangent lines (`[..., 5] Line-`) determine the unique envelope (`[...] Pole+ <- Line-`)
    touching all five."""
    first, second, third, fourth, fifth = (sides[..., index] for index in range(5))
    # Intersecting the first four lines in two pairings gives opposite vertices of their quadrilateral.
    first_corner, opposite_corner = first ^ second, third ^ fourth             # [...] Pole+, Pole+
    first_cross, opposite_cross = first ^ third, second ^ fourth               # [...] Pole+, Pole+
    # Each symmetric pole dyad is a degenerate envelope vanishing on every line through either pole.
    corner_pair = (first_corner * (opposite_corner & Line) + opposite_corner * (first_corner & Line)) / 2   # [...] Pole+ <- Line-
    cross_pair = (first_cross * (opposite_cross & Line) + opposite_cross * (first_cross & Line)) / 2        # [...] Pole+ <- Line-
    return corner_pair * (cross_pair(fifth) & fifth) - cross_pair * (corner_pair(fifth) & fifth)            # [...] Pole+ <- Line-


def tangents(inner: Envelope, points: Point) -> Point:
    """The line pencil through each exterior point in `points` (`[...] Point-`) has two tangent
    directions (`[..., 2] Point-`) at infinity to `inner` (`[...] Pole+ <- Line-`)."""
    # Joining each point with the coordinate directions at infinity spans the pencil of lines through it.
    x_direction, y_direction = mv.x.dual(), mv.y.dual()                        # [] Point-, Point-
    x_lines, y_lines = points & x_direction, points & y_direction              # [...] Line-, Line-
    xx_term = inner(x_lines) & x_lines                                         # [...] Scalar+
    xy_term = inner(y_lines) & x_lines                                         # [...] Scalar+
    yy_term = inner(y_lines) & y_lines                                         # [...] Scalar+
    # Outside `inner` the discriminant is positive and splits into the two tangent directions; on `inner` it vanishes.
    discriminant_root = (xy_term * xy_term - xx_term * yy_term).square_root()  # [...] Scalar+
    x_weights = discriminant_root[..., None] * root_signs - xy_term[..., None] # [..., 2] Scalar+
    return x_direction * x_weights + y_direction * xx_term[..., None]          # [..., 2] Point-


def path(outer: Quadric, inner: Envelope, start: Point, heading: Point, inside: Point) -> Iterator[Point]:
    """Yields successive vertices (`[...] Point-`) on `outer` by stepping along tangents of `inner`,
    using the interior point `inside` to select the outgoing tangent rationally at each vertex."""
    vertex = start.broadcast_to(heading.shape)                                 # [...] Point-
    while True:
        yield vertex
        # Along `vertex + heading * t`, `outer` vanishes at `t = 0` and at the rational second root below.
        crossing = vertex * (outer(heading) & heading) - heading * (2 * (outer(vertex) & heading))   # [...] Point-
        vertex = crossing.normalized()                                         # [...] Point-
        # Because `inside` lies strictly inside `inner`, `spoke_line` never touches `inner` and spans the pencil with `side`.
        spoke = vertex - inside                                                # [...] Point-
        side, spoke_line = vertex & heading, vertex & spoke                    # [...] Line-, Line-
        # Along `heading + spoke * t`, `inner` vanishes on the incoming `side` (`t = 0`) and on the outgoing tangent.
        outgoing = heading * (inner(spoke_line) & spoke_line) - spoke * (2 * (inner(spoke_line) & side))   # [...] Point-
        heading = outgoing / outgoing.antinorm()                               # [...] Point-


def polygon_paths(outer: Quadric, inner: Envelope, starts: Point, inside: Point,
                  n_sides: int) -> Iterator[Point]:
    """Yields the `[n_sides + 1, ...]` polygon vertices (`Point-`) around `inner` from each
    starting point in `starts`."""
    for start in starts:
        heading = tangents(inner, start)[..., 0]                               # [...] Point-
        yield stack(list(islice(path(outer, inner, start, heading, inside), n_sides + 1)))   # [n_sides + 1, ...] Point-
