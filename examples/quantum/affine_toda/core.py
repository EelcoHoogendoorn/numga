"""The local potential and three-mode interactions of affine E8 Toda theory.

The Toda field takes values in the eight-dimensional space of E8's roots, intrinsic, antivectors; the
roots are mirrors, vectors, and each reads the field by pairing, `root & field`. The potential is a sum
of exponentials of those readings, so its second and third derivatives at the vacuum are forms on the
field, sums of products of the same readings.
"""

from __future__ import annotations

import numpy as np

from numga import stack

from examples.math.e8.core import Scalar, Vector, ga

# The field's values are antivectors; the roots that read them are mirrors, vectors.
Field = ga.gatype.antivector()
Root = Vector
MassForm = ga.gatype((Scalar, Field, Field))
CubicForm = ga.gatype((Scalar, Field, Field, Field))


# --- math -----------------------------------------------------------------------------
def potential(field: Field, interaction_roots: Root, marks: np.ndarray) -> Scalar:
    """Potential energy density relative to the vacuum, with both overall scales set to one."""
    return (marks * ((interaction_roots & field[..., None]).exp() - 1)).sum(axis=-1)


# --- plumbing -------------------------------------------------------------------------
def root_triangles(roots: Root, species: np.ndarray) -> tuple[Root, Scalar]:
    """The closest-closing root triangle for each triple of species, and its squared closure."""
    n_steps = roots.shape[0]
    n_triples = len(species)
    steps = np.arange(n_steps)
    cases = np.arange(n_triples)

    # A common Coxeter turn changes no closure, so the first root can stay at step zero.
    first = roots[0, species[:, 0]]                                            # [triples] Root
    second = roots[steps[None, :], species[:, 1, None]]                         # [triples, steps] Root
    third = roots[steps[None, :], species[:, 2, None]]                          # [triples, steps] Root
    residuals = first[:, None, None] + second[:, :, None] + third[:, None, :]   # [triples, steps, steps] Root
    closure = residuals.scalar_norm_squared().reshape(n_triples, -1)           # [triples, step_pairs] Scalar
    nearest = closure.argmin(axis=-1)
    triangles = stack((first, second[cases, nearest // n_steps],
                       third[cases, nearest % n_steps]), axis=1)              # [triples, sides] Root
    return triangles, closure[cases, nearest]
