"""Nonassociative numbers drawn as the Fano plane, their Clifford actions, and fourteen multiplication-preserving
rotations."""

import numpy as np

from numga import stack
from examples.math.octonions import core

TOLERANCE = 1e-10
ANGLE = 0.8


# --- math -----------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_figure
    from examples.math.octonions import render

    save_figure(render.draw_fano(core.cycles), "octonions")
    mv, Octonion = core.mv, core.Octonion
    first, second, third = mv.a, mv.b, mv.d
    left_action = core.multiply(first, Octonion)                               # [] Octonion <- Octonion
    generators = core.symmetry_generators(TOLERANCE)                           # [generators] Bivector
    # Keep the part of an ordinary rotation that belongs to the symmetry space.
    seed = mv.ab
    symmetry = (generators * generators.reverse().scalar_product(seed)).sum(axis=0)
    rotations = (stack([seed, symmetry]) * (ANGLE / 2)).exp()                   # [cases] Rotor
    defects = core.multiplication_defect(rotations, first, third)

    print("Left multiplication by a")
    print(left_action.coefficient_table())
    print("(a b) d - a (b d)")
    print(core.associator(first, second, third).coefficient_table())
    print(f"{len(generators)} of {len(core.Bivector.output_subspace)} rotation generators preserve multiplication.")
    print("Ordinary rotation: rotate the product minus multiply the rotated factors")
    print(defects[0].coefficient_table())
    print("G₂ rotation: the same difference")
    print(defects[1].coefficient_table())

    # checks
    np.testing.assert_allclose((left_action(left_action) + Octonion).kernel, 0, atol=TOLERANCE)
    np.testing.assert_allclose((core.associator(first, second, third) - 2 * mv.g).kernel, 0, atol=TOLERANCE)
    np.testing.assert_allclose(defects[1].kernel, 0, atol=TOLERANCE)
    assert core.norm_squared(defects[0]).kernel.item() > 0.1


if __name__ == "__main__":
    main()
