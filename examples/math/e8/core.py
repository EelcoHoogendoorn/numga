"""Generates the 240 roots of E8 and the four orthogonal planes of its Coxeter rotation.

Eight simple-root mirrors generate the reflections of E8, and their geometric product forms a rotor whose
action on vectors carries eight reflection seeds around disjoint orbits of thirty roots. Averaging that
orthogonal action with its adjoint isolates the four invariant rotation planes and their integer winding
rates in one symmetric eigensolve.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import Algebra, NumpyContext
from numga.extensor import stack

ga = Algebra("a+b+c+d+e+f+g+h+")
context = NumpyContext(ga)
mv = context.multivector
Scalar = ga.gatype.scalar()                                                   # Scalar+
Vector = ga.gatype.vector()                                                   # Vector+
Bivector = ga.gatype.bivector()                                               # Bivector+
Rotor = ga.gatype.rotor()                                                     # Rotor+
VectorMap = ga.gatype((Vector, Vector))                                       # Vector+ <- Vector+
coxeter_order = 30


# --- plumbing -------------------------------------------------------------------------
def simple_roots() -> Vector:
    """Eight simple-root mirrors (`[8] Vector+`) of squared norm two generate the reflection group of E8."""
    return stack([
        (mv.a + mv.h - mv.b - mv.c - mv.d - mv.e - mv.f - mv.g) / 2,
        mv.a + mv.b,
        mv.b - mv.a,
        mv.c - mv.b,
        mv.d - mv.c,
        mv.e - mv.d,
        mv.f - mv.e,
        mv.g - mv.f,
    ])                                                                        # [8] Vector+


# --- math -----------------------------------------------------------------------------
def reflection_seeds(roots: Vector) -> Iterator[Vector]:
    """Yields one seed root (`[] Vector+`) per Coxeter orbit by reflecting each mirror in the mirrors before it."""
    prefix = Vector                                                           # [] Vector+ <- Vector+
    for root in roots:
        yield prefix(root)                                                    # [] Vector+
        # Composing on the right applies earlier mirrors after later ones, matching the Coxeter product order.
        reflection = root.normalized() >> Vector                              # [] Vector+ <- Vector+
        prefix = prefix(reflection)                                           # [] Vector+ <- Vector+


def orbit(action: VectorMap, seeds: Vector, n_steps: int) -> Iterator[Vector]:
    """Yields `n_steps` successive images (`[...] Vector+`) of `seeds` under `action`."""
    for _ in range(n_steps):
        yield seeds                                                           # [...] Vector+
        seeds = action(seeds)                                                 # [...] Vector+


def invariant_planes(action: VectorMap, n_steps: int) -> tuple[Vector, Vector, Bivector, Scalar]:
    """Symmetric eigendecomposition of `action` isolates the four invariant plane frames, bivectors, and turn counts (`[4]`)."""
    symmetric = (action + action.adjoint()) / 2                               # [] Vector+ <- Vector+
    cosines, eigenvectors = symmetric.eigh()                                  # [8] Scalar+, Vector+
    # Equal eigenvalues occur in pairs; descending cosines order the four planes from slow to fast.
    plane_x = eigenvectors[::-2]                                              # [4] Vector+
    # Subtracting the projection along plane_x isolates the perpendicular turn direction in each plane.
    plane_y = (action(plane_x) - cosines[::-2] * plane_x).normalized()        # [4] Vector+
    planes = plane_x ^ plane_y                                                # [4] Bivector+
    rates = cosines[::-2].arccos() * n_steps / (2 * np.pi)                    # [4] Scalar+
    return plane_x, plane_y, planes, rates


def step_rotor(planes: Bivector, rates: Scalar, n_steps: int, n_substeps: int) -> Rotor:
    """Commuting plane generators (`[4] Bivector+`) exponentiate to the fractional-step rotor (`[] Rotor+`)."""
    generator = -(planes * rates).sum(axis=0) * np.pi / n_steps               # [] Bivector+
    return (generator / n_substeps).exp()                                     # [] Rotor+
