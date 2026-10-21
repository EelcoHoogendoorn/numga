"""E8 Toda masses from the vacuum response, and cubic couplings from root triangles."""

from __future__ import annotations

from itertools import combinations_with_replacement

import numpy as np

from numga import concatenate

from examples.math.e8 import core as e8
from examples.math.e8.scenarios import root_system
from examples.quantum.affine_toda import core
from examples.quantum.affine_toda.core import CubicForm, Field, MassForm, Root, Scalar

# The highest root's coefficients on the simple roots, followed by the weight of its negative.
marks = np.array([2, 3, 4, 6, 5, 4, 3, 2, 1])


# --- math -----------------------------------------------------------------------------
def spectrum() -> tuple[Root, np.ndarray, MassForm, CubicForm, Scalar, Field, Root, Root, Root]:
    """Interaction roots, vacuum curvature, masses and modes, and mass-ordered Coxeter orbits."""
    roots, plane_x, plane_y, _, _, _, _ = root_system()                        # [steps, orbits] Root
    simple_roots = e8.simple_roots()                                          # [simple_roots] Root

    # The negative highest root closes the weighted root sum, making zero field an equilibrium.
    affine_root = -(marks[:-1] * simple_roots).sum(axis=0)                      # [] Root
    interaction_roots = concatenate((simple_roots, affine_root[None]))         # [interactions] Root

    # Each exponential contributes the product of its root's readings to the vacuum's derivatives.
    reading = interaction_roots & Field                                       # [interactions] Scalar <- Field
    curvature = (marks * reading * reading).sum(axis=0)                        # [] Scalar <- (Field, Field)
    cubic = (marks * reading * reading * reading).sum(axis=0)                  # [] Scalar <- (Field, Field, Field)
    # With a canonical kinetic term, the curvature's eigenvalues against the field's metric are squared masses.
    eigenvalues, modes = curvature.eigh()                                     # [species] Scalar, Field
    masses = eigenvalues.square_root()                                        # [species] Scalar

    # The smallest-turning plane gives eight radii in the same ratios as the eight masses.
    radius_squared = (roots[0] | plane_x[0]).squared() + (roots[0] | plane_y[0]).squared()
    roots = roots[:, radius_squared.argsort()]                                # [steps, species] Root
    return interaction_roots, marks, curvature, cubic, masses, modes, roots, plane_x[0], plane_y[0]


def interactions() -> tuple[np.ndarray, Root, Scalar, Scalar, Scalar]:
    """Every unordered species triple, its closest root triangle, coupling and projected area."""
    _, _, _, cubic, masses, modes, roots, plane_x, plane_y = spectrum()
    n_species = len(masses)
    species = np.array(list(combinations_with_replacement(range(n_species), 3)))
    triangles, closure = core.root_triangles(roots, species)                   # [triples, sides] Root; [triples] Scalar
    couplings = cubic(modes[species[:, 0]], modes[species[:, 1]],
                      modes[species[:, 2]])                                   # [triples] Scalar

    # Triangle areas in the Coxeter plane are measured by its unit oriented bivector.
    plane = plane_x ^ plane_y                                                 # [] Bivector
    areas = (triangles[:, 0] ^ triangles[:, 1]).scalar_product(plane).abs() / 2 # [triples] Scalar
    return species, triangles, closure, couplings, areas


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.animation import save_figure
    from examples.quantum.affine_toda import render

    species = np.array([[0, 0, 1], [0, 1, 2], [1, 2, 4]])
    _, _, _, cubic, masses, modes, roots, horizontal, vertical = spectrum()
    radii = ((roots[0] | horizontal).squared()
             + (roots[0] | vertical).squared()).square_root()                  # [species] Scalar
    triangles, closure = core.root_triangles(roots, species)                   # [cases, sides] Root; [cases] Scalar
    couplings = cubic(modes[species[:, 0]], modes[species[:, 1]],
                      modes[species[:, 2]])                                   # [cases] Scalar
    paths = concatenate((0 * triangles[:, :1], triangles.cumsum(axis=1)), axis=1) # [cases, corners] Root

    print(render.mass_table(masses, radii))
    save_figure(render.draw_roots(roots, horizontal, vertical), "affine_toda_masses")
    print(render.coupling_table(species, couplings, closure))
    save_figure(render.draw_triangles(paths, roots, species, horizontal, vertical), "affine_toda_triangles")

    # checks
    # The masses stand in the ratios of the Coxeter rings' radii, and the first two triples close their
    # root triangles and couple while the third does neither.
    np.testing.assert_allclose((radii / radii[0] - masses / masses[0]).to_array(), 0, atol=1e-10)
    np.testing.assert_array_equal(closure.to_array() < 1e-10, [True, True, False])
    np.testing.assert_array_equal(couplings.abs().to_array() > 1e-10, [True, True, False])


if __name__ == "__main__":
    main()
