"""Four lines in space, and the two lines that meet all four.

A line in space is a bivector of PGA3D, but most bivectors are not lines: a bivector is a line when
its wedge with itself vanishes, and two lines meet when their wedge vanishes. Both are one form,
`KLEIN = Bivector ^ Bivector`, a quadratic form on the six-dimensional space of bivectors. The lines
are its zeros, the Klein quadric, and two lines meet when the form pairs them to zero.

Meeting a given line is one linear condition on a bivector, `line ^ bivector == 0`. Four lines leave
a pencil of bivectors that meet all four: the combinations of two of them. Along the pencil the
Klein form is a quadratic with two zeros, so two bivectors of the pencil are lines, the two
transversals. With complex coefficients the two always exist; for real lines they are either both
real or a complex-conjugate pair. The algebra here has complex coefficients, so that the pair is
there in every case.

In space the same count reads differently. The lines that meet three of the four sweep out a ruled
quadric, here a hyperboloid; the fourth line crosses it at two points, and through each passes the
line of the sweep that meets the fourth line too.

In Plücker coordinates a line reads as six numbers on the Klein quadric in five-dimensional
projective space, and in the Schubert calculus of the Grassmannian of lines the count of two reads
as the intersection number of four Schubert cycles.
"""

from __future__ import annotations

import numpy as np

from numga import NumpyContext, stack
from numga.algebras import PGA3D as ga

mv = NumpyContext(ga, np.complex128).multivector
Scalar = ga.gatype.scalar()
Plane = ga.gatype.vector()
Bivector = ga.gatype.bivector()
Point = ga.gatype.antivector()
Quadric = ga.gatype((Plane, Point))
# Zero on one bivector twice exactly when it is a line; zero on two lines exactly when they meet.
KLEIN = Bivector ^ Bivector                                                    # [] Pseudoscalar <- (Bivector, Bivector)
# The origin, the point dual to the plane at infinity.
ORIGIN = mv.w.dual()                                                           # [] Point
# Both roots of a quadratic, one per sign.
SIGNS = np.array([1.0, -1.0])


# --- math -----------------------------------------------------------------------------
def roots(a: Scalar, b: Scalar, c: Scalar) -> Scalar:
    """The two roots of the quadratic `a + 2 * b * t + c * t * t` along the combinations of two ends,
    as the weights of the ends on the last axis, one root per sign on the axis before it:
    `first * (sign * root - b) + second * a`, with `root * root == b * b - a * c`."""
    root = (b * b - a * c).square_root()                                       # [...] Scalar
    first = root[..., None] * SIGNS - b[..., None]                             # [..., 2] Scalar
    return stack([first, a[..., None].broadcast_to(first.shape)], axis=-1)     # [..., 2, 2] Scalar


def transversals(lines: Bivector) -> Bivector:
    """The two lines that meet all four lines on the last axis, at unit norm, on the last axis."""
    # Each line, weighted by the Klein form between it and a bivector: zero exactly on the bivectors
    # that meet all four.
    misses = (lines * KLEIN(lines).dual()).sum(axis=-1)                        # [...] Bivector <- Bivector
    # Those bivectors form a pencil, spanned by the right singular vectors of zero singular value.
    _, _, right = misses.svd()
    pencil = right[..., -2:]                                                   # [..., 2] Bivector
    # Along the pencil the Klein form is a quadratic; its two zeros are the lines of the pencil.
    first, second = pencil[..., 0], pencil[..., 1]                             # [...] Bivector
    weights = roots(KLEIN(first, first).dual(), KLEIN(first, second).dual(), KLEIN(second, second).dual())   # [..., 2, 2] Scalar
    return (weights * pencil[..., None, :]).sum(axis=-1).normalized()          # [..., 2] Bivector


def ruled_quadric(first: Bivector, second: Bivector, third: Bivector) -> Quadric:
    """The quadric swept by the lines that meet three lines, as a map from points to planes that
    pairs to zero with the points on it. The plane joining a point to the first line meets the third
    line in a point; the plane joining that point to the second line holds the first point exactly
    when a line through it meets all three."""
    return ((Point & first) ^ third) & second                                  # [...] Plane <- Point


def crossings(surface: Quadric, line: Bivector) -> Point:
    """The two points where a line crosses a quadric, at unit weight, on the last axis."""
    # The line's point nearest the origin and its point at infinity span it.
    span = stack([(ORIGIN | line) ^ line, line ^ mv.w], axis=-1)               # [..., 2] Point
    first, second = span[..., 0], span[..., 1]                                 # [...] Point
    # The quadric's map need not be symmetric: the cross term is the mean of both orders.
    cross = ((surface(first) & second) + (surface(second) & first)) / 2        # [...] Scalar
    weights = roots(surface(first) & first, cross, surface(second) & second)   # [..., 2, 2] Scalar
    crossing = (weights * span[..., None, :]).sum(axis=-1)                     # [..., 2] Point
    return crossing / (mv.w & crossing)                                        # [..., 2] Point


def through(points: Point, second: Bivector, third: Bivector) -> Bivector:
    """The line through each point that meets two lines: the meet of the planes joining the point to
    each line."""
    return (points & second) ^ (points & third)                                # [...] Bivector


# --- plumbing -------------------------------------------------------------------------
def point(coords: np.ndarray) -> Point:
    """Points at unit weight from their x, y and z coordinates."""
    return (mv("x y z", coords) + mv.w).dual()


def spread(line: Bivector, count: int) -> Point:
    """Points spread over a whole line, its point at infinity included: the unit-weight point nearest
    the origin turned towards the unit direction by angles spaced evenly over half a turn."""
    unit = line.normalized()                                                   # [...] Bivector
    nearest = (ORIGIN | unit) ^ unit                                           # [...] Point
    angles = np.linspace(0.0, np.pi, count, endpoint=False)
    return (nearest / (mv.w & nearest))[..., None] * np.cos(angles) + (unit ^ mv.w)[..., None] * np.sin(angles)   # [..., count] Point
