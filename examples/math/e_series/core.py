"""The exceptional root systems E6, E7 and E8 from nested sets of mirrors, in the geometric algebra of
eight dimensions.

The first six, seven or eight of E8's simple roots generate E6, E7 or E8. Each system's reflections,
composed as maps on vectors, are its Coxeter motion, whose orbits through one seed each hold all its
roots; averaged with its adjoint, the motion gives its planes of turning by one eigensolve.
"""

from __future__ import annotations

from functools import reduce

from numga import stack

from examples.math.e8.core import Scalar, Vector, VectorMap, orbit, reflection_seeds, simple_roots


# --- math -----------------------------------------------------------------------------
def root_system(simple: Vector, n_steps: int) -> tuple[VectorMap, Vector]:
    """The Coxeter motion, the generating reflections composed, and the roots it carries one seed from each
    orbit through."""
    reflections = simple.normalized() >> Vector                               # [mirrors] Vector+ <- Vector+
    # Composing from the right matches the order of the prefixes that pick the seeds.
    action = reduce(lambda left, right: left(right), reflections)             # [] Vector+ <- Vector+
    seeds = stack(reflection_seeds(simple))                                   # [orbits] Vector+
    roots = stack(orbit(action, seeds, n_steps))                              # [steps, orbits] Vector+
    return action, roots


def rotation_planes(action: VectorMap, n_rank: int) -> tuple[Vector, Vector, Scalar]:
    """The orthogonal rotation planes of a Coxeter motion, from slowest to fastest."""
    symmetric = (action + action.adjoint()) / 2                               # [] Vector+ <- Vector+
    cosines, eigenvectors = symmetric.eigh()                                  # [eigenvectors] Scalar+, Vector+

    # The eigenvectors outside the root span have cosine one and sort last.
    # Paired cosines describe turns; an unpaired minus one describes a reflection.
    plane_cosines = cosines[n_rank - 1:0:-2]                                  # [planes] Scalar+
    plane_x = eigenvectors[n_rank - 1:0:-2]                                   # [planes] Vector+
    plane_y = (action(plane_x) - plane_cosines * plane_x).normalized()        # [planes] Vector+
    return plane_x, plane_y, plane_cosines.arccos()
