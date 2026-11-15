"""Four lines in space in general position are met by two transversals, found here through the Klein
quadric.

A line in space is modelled as a bivector of projective geometric algebra, where a bivector represents
a line when its wedge with itself vanishes and two lines meet when their wedge vanishes. Both
conditions come from one quadratic form on the six-dimensional space of bivectors, `klein = Bivector ^
Bivector`, whose zeros form the Klein quadric.

Meeting a given line imposes one linear condition `line ^ bivector == 0` on a bivector. Four lines in
general position leave a pencil of bivectors meeting all four, and restricting `klein` to that pencil
gives a quadratic with two zeros, the two transversals. Over complex coefficients both zeros always
exist: for real lines they are either two real lines or a complex-conjugate pair.

In three-dimensional space the same count appears on a ruled surface. The lines meeting three of the
four given lines sweep out a ruled quadric, here a hyperboloid of one sheet; the fourth line crosses
that quadric at two points, and through each crossing passes the ruling that meets all four lines.
"""

from __future__ import annotations

import numpy as np

from numga import NumpyContext
from numga.algebras import PGA3D as ga

mv = NumpyContext(ga, np.complex128).multivector
Scalar = ga.gatype.scalar()                                                    # Scalar+
# Every line here is a meet of two planes, an extrinsic bivector axis.
Bivector = ga.gatype.bivector()                                                # Bivector+
Point = ga.gatype.antivector()                                                 # Antivector-
# Joining a point to a line and meeting with two others maps points to intrinsic planes.
Quadric = ga.gatype((ga.gatype.antitrivector(), Point))                        # Antitrivector- <- Antivector-
# Vanishes on one bivector twice when it is a line, and on two lines when they meet.
klein = Bivector ^ Bivector                                                    # [] Pseudoscalar+ <- (Bivector+, Bivector+)
# The origin is dual to the plane at infinity, whose join of three directions meets each line at infinity.
origin = mv.w.dual()                                                           # [] Antivector-
horizon = mv.direction([1, 0, 0]) & mv.direction([0, 1, 0]) & mv.direction([0, 0, 1])   # [] Antitrivector-
# Signs selecting the two roots of a quadratic.
signs = np.array([1, -1])


# --- math -----------------------------------------------------------------------------
def quadratic_roots(half_linear: Scalar, constant: Scalar) -> Scalar:
    """The monic quadratic `t * t + 2 * half_linear * t + constant` has two roots (`[..., 2]
    Scalar+`), giving the weights on the first spanning element against unit weight on the second."""
    discriminant_root = (half_linear * half_linear - constant).square_root()   # [...] Scalar+
    return discriminant_root[..., None] * signs - half_linear[..., None]       # [..., 2] Scalar+


def transversals_of(lines: Bivector) -> Bivector:
    """Four lines on the trailing axis (`[..., 4] Bivector+`) determine two unit-norm transversals
    (`[..., 2] Bivector+`) meeting all four."""
    # Pairing a candidate bivector against all four lines maps the pencil of common meets to zero.
    incidence_map = (lines * klein(lines).dual()).sum(axis=-1)                 # [...] Bivector- <- Bivector+
    # The two right singular vectors of zero singular value span that pencil of bivectors.
    _, _, right_vectors = incidence_map.svd()
    pencil = right_vectors[..., -2:]                                           # [..., 2] Bivector+
    # Along the pencil the Klein form reduces to a scalar quadratic whose two zeros are genuine lines.
    first_end, second_end = pencil[..., 0], pencil[..., 1]                     # [...] Bivector+, Bivector+
    leading = klein(first_end, first_end).dual()                               # [...] Scalar-
    weights = quadratic_roots(                                                 # [..., 2] Scalar+
        klein(first_end, second_end).dual() / leading,
        klein(second_end, second_end).dual() / leading,
    )
    return (first_end[..., None] * weights + second_end[..., None]).normalized()   # [..., 2] Bivector+


def crossings_of(surface: Quadric, line: Bivector) -> Point:
    """A line (`[...] Bivector+`) crosses `surface` at two unit-weight points (`[..., 2]
    Antivector-`)."""
    # Every finite line is spanned by its point nearest the origin and its ideal direction point.
    nearest = (origin | line) ^ line                                           # [...] Antivector-
    direction = line ^ horizon                                                 # [...] Antivector-
    # Averaging both orders extracts the symmetric cross term even when `surface` is unsymmetrized.
    cross = ((surface(nearest) & direction) + (surface(direction) & nearest)) / 2   # [...] Scalar-
    leading = surface(nearest) & nearest                                       # [...] Scalar-
    weights = quadratic_roots(cross / leading, (surface(direction) & direction) / leading)   # [..., 2] Scalar+
    crossings = nearest[..., None] * weights + direction[..., None]            # [..., 2] Antivector-
    return crossings.normalized()                                              # [..., 2] Antivector-


def line_points(line: Bivector, n_points: int) -> Point:
    """Homogeneous combinations of the unit-weight point nearest the origin and the unit ideal
    direction sample `n_points` points (`[..., n_points] Antivector-`) around the full projective
    line."""
    unit_line = line.normalized()                                              # [...] Bivector+
    nearest = ((origin | unit_line) ^ unit_line).normalized()                  # [...] Antivector-
    direction = unit_line ^ horizon                                            # [...] Antivector-
    angles = np.linspace(0.0, np.pi, n_points, endpoint=False)
    return nearest[..., None] * np.cos(angles) + direction[..., None] * np.sin(angles)   # [..., n_points] Antivector-
