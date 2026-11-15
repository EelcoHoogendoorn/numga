"""Affine E8 Toda theory has a local potential whose derivatives at the vacuum give its masses and three-mode interactions.

The Toda field takes values in the eight-dimensional space of E8's roots as antivectors; the roots are mirrors, vectors, and each reads the field by pairing, `root & field`. The potential is a weighted sum of exponentials of those readings, so its second and third derivatives at the vacuum are forms on the field built from products of the same readings.
"""

from __future__ import annotations

import numpy as np

from numga import concatenate, stack

from examples.math.e8 import core as e8
from examples.math.e8.core import Bivector, Scalar, Vector, ga

# The field's values are antivectors; the roots that read them are mirrors, vectors.
Field = ga.gatype.antivector()  # Antivector-
Root = Vector                   # Vector+
MassForm = ga.gatype((Scalar, Field, Field))
CubicForm = ga.gatype((Scalar, Field, Field, Field))


# --- math -----------------------------------------------------------------------------
def interaction_roots(simple_roots: Root, marks: np.ndarray) -> Root:
    """The eight simple roots and the negative highest root (`[9] Root+`) have a vanishing marked sum at the vacuum."""
    affine_root = -(marks[:-1] * simple_roots).sum(axis=0)                      # [] Root+
    return concatenate((simple_roots, affine_root[None]))                       # [9] Root+


def potential(field: Field, roots: Root, marks: np.ndarray) -> Scalar:
    """The marked Toda potential density relative to the vacuum (`[...] Scalar+`) sums each root's exponentiated reading less one."""
    return (marks * ((roots & field[..., None]).exp() - 1)).sum(axis=-1)        # [...] Scalar+


def vacuum_forms(roots: Root, marks: np.ndarray) -> tuple[MassForm, CubicForm]:
    """The marked products of root readings give the quadratic mass form and cubic interaction form on the field."""
    reading = roots & Field                                                     # [9] Scalar+ <- Field-
    curvature = (marks * reading * reading).sum(axis=0)                         # [] Scalar+ <- (Field-, Field-)
    cubic = (marks * reading * reading * reading).sum(axis=0)                   # [] Scalar+ <- (Field-, Field-, Field-)
    return curvature, cubic


def mass_spectrum(curvature: MassForm, root_orbits: Root,
                  coxeter_plane: Bivector) -> tuple[Scalar, Field, Root, Scalar]:
    """The vacuum curvature's eigenvalues give the eight particle masses (`[8] Scalar+`), ordered alongside the Coxeter ring radii."""
    eigenvalues, modes = curvature.eigh()                                       # [8] Scalar+, [8] Field-
    masses = eigenvalues.square_root()                                          # [8] Scalar+
    radii = (root_orbits[0] | coxeter_plane).scalar_norm_squared().square_root()  # [8] Scalar+
    order = radii.argsort()
    return masses, modes, root_orbits[:, order], radii[order]


def area_couplings(triangles: Root, mass_scale: Scalar, coxeter_plane: Bivector) -> Scalar:
    """Projected wedge areas of mass-scaled root triangles in the Coxeter plane determine the cubic coupling magnitudes (`[...] Scalar+`)."""
    mass_triangles = triangles * mass_scale                                     # [..., 3] Root+
    areas = (mass_triangles[:, 0] ^ mass_triangles[:, 1]).scalar_product(coxeter_plane).abs() / 2  # [...] Scalar+
    return 4 * areas / np.sqrt(e8.coxeter_order)                                # [...] Scalar+


# --- plumbing -------------------------------------------------------------------------
def root_triangles(roots: Root, species: np.ndarray) -> tuple[Root, Scalar]:
    """Each species triple selects a closest-closing root triangle (`[n_triples, 3] Root+`) with squared closure gap (`[n_triples] Scalar+`)."""
    n_steps = roots.shape[0]
    n_triples = len(species)
    step_indices = np.arange(n_steps)
    triple_indices = np.arange(n_triples)

    # A common Coxeter turn preserves closure, so the first root stays at step zero.
    first_roots = roots[0, species[:, 0]]                                       # [n_triples] Root+
    second_roots = roots[step_indices[None, :], species[:, 1, None]]            # [n_triples, n_steps] Root+
    third_roots = roots[step_indices[None, :], species[:, 2, None]]             # [n_triples, n_steps] Root+
    residuals = first_roots[:, None, None] + second_roots[:, :, None] + third_roots[:, None, :]  # [n_triples, n_steps, n_steps] Root+
    closure = residuals.scalar_norm_squared().reshape(n_triples, -1)            # [n_triples, n_steps * n_steps] Scalar+
    nearest = closure.argmin(axis=-1)
    triangles = stack((first_roots, second_roots[triple_indices, nearest // n_steps],
                       third_roots[triple_indices, nearest % n_steps]), axis=1)  # [n_triples, 3] Root+
    return triangles, closure[triple_indices, nearest]
