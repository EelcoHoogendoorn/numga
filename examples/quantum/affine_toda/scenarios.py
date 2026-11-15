"""The E8 Toda mass spectrum follows from the vacuum curvature, and its cubic couplings follow from root triangles."""

from __future__ import annotations

from itertools import combinations_with_replacement

import numpy as np

from examples.math.e8 import core as e8
from examples.math.e8.scenarios import root_system
from examples.quantum.affine_toda import core
from examples.quantum.affine_toda.core import CubicForm, Field, MassForm, Root, Scalar

# The highest root's coefficients on the simple roots, followed by the unit weight of its negative.
marks = np.array([2, 3, 4, 6, 5, 4, 3, 2, 1])
# IN12 peak positions in meV from Gaussian fits to neutron-scattering data in BaCo2V2O8.
measured_energies = np.array([1.26, 2.05, 2.49])
# Three illustrative species triples: 1-1-2 and 1-2-3 close and couple, whereas 2-3-5 does neither.
example_species = np.array([[0, 0, 1], [0, 1, 2], [1, 2, 4]])


def spectrum() -> tuple[Root, np.ndarray, MassForm, CubicForm, Scalar, Field, Root, Scalar, Root, Root]:
    """The nine interaction roots determine the vacuum forms, particle masses, eigenmodes, and radius-ordered Coxeter orbits."""
    root_orbits, plane_x, plane_y, planes, *_ = root_system()                   # [30, 8] Root+, [4] Root+, [4] Root+, [4] Bivector+
    roots = core.interaction_roots(e8.simple_roots(), marks)                    # [9] Root+
    curvature, cubic = core.vacuum_forms(roots, marks)                          # [] Scalar+ <- (Field-, Field-), [] Scalar+ <- (Field-, Field-, Field-)
    masses, modes, root_orbits, radii = core.mass_spectrum(curvature, root_orbits, planes[0])  # [8] Scalar+, [8] Field-, [30, 8] Root+, [8] Scalar+
    return roots, marks, curvature, cubic, masses, modes, root_orbits, radii, plane_x[0], plane_y[0]


def interactions() -> tuple[np.ndarray, Root, Scalar, Scalar, Scalar]:
    """Every unordered triple of the eight species has a closest root triangle, cubic coupling, and area-predicted magnitude."""
    _, _, _, cubic, masses, modes, roots, radii, plane_x, plane_y = spectrum()
    n_species = len(masses)
    species = np.array(list(combinations_with_replacement(range(n_species), 3)))
    n_triples = len(species)
    triangles, closure = core.root_triangles(roots, species)                    # [n_triples, 3] Root+, [n_triples] Scalar+
    couplings = cubic(modes[species[:, 0]], modes[species[:, 1]],
                      modes[species[:, 2]])                                     # [n_triples] Scalar+
    coxeter_plane = plane_x ^ plane_y                                           # [] Bivector+
    predicted = core.area_couplings(triangles, masses[0] / radii[0], coxeter_plane)  # [n_triples] Scalar+
    return species, triangles, closure, couplings, predicted


def main() -> None:
    from examples.quantum.affine_toda import render
    from examples.util.animation import save_figure

    n_cases = len(example_species)
    _, _, _, cubic, masses, modes, roots, radii, horizontal, vertical = spectrum()
    coxeter_plane = horizontal ^ vertical                                       # [] Bivector+
    triangles, closure = core.root_triangles(roots, example_species)            # [n_cases, 3] Root+, [n_cases] Scalar+
    couplings = cubic(modes[example_species[:, 0]], modes[example_species[:, 1]],
                      modes[example_species[:, 2]])                             # [n_cases] Scalar+
    closed = closure < 1e-6
    predicted = core.area_couplings(triangles[closed], masses[0] / radii[0], coxeter_plane)  # [...] Scalar+

    print("masses over the lightest:", (masses / masses[0]).to_array())
    print("radii over the smallest:", (radii / radii[0]).to_array())
    save_figure(render.draw_roots(roots, horizontal, vertical), "affine_toda_masses")

    print("species:", example_species + 1)
    print("|coupling|:", couplings.abs().to_array())
    print("root closure:", closure.square_root().to_array())
    save_figure(render.triangles_figure(triangles, roots, example_species, horizontal, vertical), "affine_toda_triangles")

    # checks
    # Mass ratios match the Coxeter ring radii, the first two triples close and couple while the third does neither,
    # and the cubic couplings of closed triangles match their scaled Coxeter-plane areas.
    assert ((radii / radii[0] - masses / masses[0]).abs() < 1e-8).all()
    np.testing.assert_array_equal(closed, [True, True, False])
    np.testing.assert_array_equal(couplings.abs() > 1e-6, [True, True, False])
    assert ((couplings[closed].abs() - predicted).abs() < 1e-8).all()


if __name__ == "__main__":
    main()
