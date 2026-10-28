"""Four lines in space in general position are met by exactly two lines, found here through the Klein
quadric.

A line in space is modelled as a bivector of PGA3D, but most bivectors are not lines: a bivector is
a line when its wedge with itself vanishes, and two lines meet when their wedge vanishes. Both are
one form, `klein = Bivector ^ Bivector`, a quadratic form on the six-dimensional space of bivectors.
The lines are its zeros, the Klein quadric, and two lines meet when the form pairs them to zero.

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

from numga import NumpyContext
from numga.algebras import PGA3D as ga

mv = NumpyContext(ga, np.complex128).multivector
Scalar = ga.gatype.scalar()                                                    # Scalar+
# Every line here is a meet of two planes, as the lines through a point that meet two lines are: an axis.
Bivector = ga.gatype.bivector()                                                # Bivector+
Point = ga.gatype.antivector()                                                 # Antivector-
# A point joined with a line and met with others gives a plane through points: oriented within itself.
Quadric = ga.gatype((ga.gatype.antitrivector(), Point))
# Zero on one bivector twice exactly when it is a line; zero on two lines exactly when they meet.
klein = Bivector ^ Bivector                                                    # [] Pseudoscalar+ <- (Bivector+, Bivector+)
# The origin, the point dual to the plane at infinity; the plane at infinity as the join of the three
# directions, whose meet with an axis is the axis's point at infinity.
origin = mv.w.dual()                                                           # [] Point-
horizon = mv.direction([1, 0, 0]) & mv.direction([0, 1, 0]) & mv.direction([0, 0, 1])   # [] Plane-
# Both roots of a quadratic, one per sign.
signs = np.array([1, -1])


# --- math -----------------------------------------------------------------------------
def roots(b: Scalar, c: Scalar) -> Scalar:
    """The two roots of the quadratic `t * t + 2 * b * t + c` along the combinations of two ends,
    as the weight of the first end against unit weight on the second, one root per sign on the last
    axis: `first * (sign * root - b) + second`, with `root * root == b * b - c`. A quadratic whose
    coefficients share a factor, such as the handedness of a signed volume, is divided by its first
    coefficient to read as this one."""
    root = (b * b - c).square_root()                                           # [...] Scalar+
    return root[..., None] * signs - b[..., None]                              # [..., 2] Scalar+


def transversals_of(lines: Bivector) -> Bivector:
    """The two lines that meet all four lines on the last axis, at unit norm, on the last axis."""
    # Each line, weighted by the Klein form between it and a bivector: zero exactly on the bivectors
    # that meet all four.
    misses = (lines * klein(lines).dual()).sum(axis=-1)                        # [...] Bivector- <- Bivector+
    # Those bivectors form a pencil, spanned by the right singular vectors of zero singular value.
    _, _, right = misses.svd()
    pencil = right[..., -2:]                                                   # [..., 2] Bivector+
    # Along the pencil the Klein form is a quadratic; its two zeros are the lines of the pencil.
    first, second = pencil[..., 0], pencil[..., 1]                             # [...] Bivector+, Bivector+
    # The Klein form of two axes is a signed volume; its ratios are ordinary numbers.
    leading = klein(first, first).dual()                                       # [...] Scalar-
    weights = roots(klein(first, second).dual() / leading, klein(second, second).dual() / leading)   # [..., 2] Scalar+
    return (first[..., None] * weights + second[..., None]).normalized()       # [..., 2] Bivector+


def crossings_of(surface: Quadric, line: Bivector) -> Point:
    """The two points where a line crosses a quadric, at unit weight, on the last axis."""
    # The line's point nearest the origin and its point at infinity, its direction, span it.
    nearest = (origin | line) ^ line                                           # [...] Point-
    direction = line ^ horizon                                                 # [...] Point-
    # The quadric's map need not be symmetric: the cross term is the mean of both orders.
    cross = ((surface(nearest) & direction) + (surface(direction) & nearest)) / 2   # [...] Scalar-
    leading = surface(nearest) & nearest                                       # [...] Scalar-
    weights = roots(cross / leading, (surface(direction) & direction) / leading)   # [..., 2] Scalar+
    crossings = nearest[..., None] * weights + direction[..., None]            # [..., 2] Point-
    return crossings.normalized()                                              # [..., 2] Point-


# --- plumbing -------------------------------------------------------------------------
def spread(line: Bivector, n_points: int) -> Point:
    """Points spread over a whole line, its point at infinity included: the unit-weight point nearest
    the origin turned towards the unit direction by angles spaced evenly over half a turn."""
    unit = line.normalized()                                                   # [...] Bivector+
    nearest = ((origin | unit) ^ unit).normalized()                            # [...] Point-
    angles = np.linspace(0.0, np.pi, n_points, endpoint=False)
    return nearest[..., None] * np.cos(angles) + (unit ^ horizon)[..., None] * np.sin(angles)   # [..., points] Point-
