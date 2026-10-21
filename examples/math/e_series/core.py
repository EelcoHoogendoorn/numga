"""The exceptional root systems E6, E7 and E8 from nested sets of mirrors, in the geometric algebra of
eight dimensions.

The first six, seven or eight of E8's simple roots generate E6, E7 or E8. Each system's reflections,
composed as maps on vectors, are its Coxeter motion, whose orbits through one seed each hold all its
roots; averaged with its adjoint, the motion gives its planes of turning by one eigensolve.
"""

from __future__ import annotations

from functools import reduce

from numga import stack

from examples.math.e8.core import (
    Bivector, Scalar, Vector, VectorMap, ga, mv,
    orbit, reflection_seeds, simple_roots,
)


# --- math -----------------------------------------------------------------------------
def root_system(simple: Vector, n_steps: int) -> tuple[VectorMap, Vector]:
    """Compose the generating reflections and follow one seed from each root orbit."""
    reflections = simple.normalized() >> Vector                              # [mirrors] Vector <- Vector
    action = reduce(lambda left, right: left(right), reflections)              # Vector <- Vector
    seeds = stack(reflection_seeds(simple))                                   # [orbits] Vector
    roots = stack(orbit(action, seeds, n_steps))                                # [steps, orbits] Vector
    return action, roots


def rotation_planes(action: VectorMap, n_rank: int) -> tuple[Vector, Vector, Scalar]:
    """The orthogonal rotation planes of a Coxeter motion, from slowest to fastest."""
    symmetric = (action + action.adjoint()) / 2                               # Vector <- Vector
    cosines, directions = symmetric.eigh()                                    # [directions] Scalar, Vector

    # The unused directions of the ambient space have cosine one and sort last.
    # Paired cosines describe turns; an unpaired minus one describes a reflection.
    plane_cosines = cosines[n_rank - 1:0:-2]                                  # [planes] Scalar
    horizontal = directions[n_rank - 1:0:-2]                                  # [planes] Vector
    vertical = (action(horizontal) - plane_cosines * horizontal).normalized()  # [planes] Vector
    return horizontal, vertical, plane_cosines.arccos()
