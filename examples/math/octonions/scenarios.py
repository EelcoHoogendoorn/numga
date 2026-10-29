"""Nonassociative numbers drawn as the Fano plane, their Clifford actions, and fourteen multiplication-preserving
rotations."""

import numpy as np

from numga import stack
from examples.math.octonions import core

# The singular values counted as zero, and the angle of the finite turns.
tolerance = 1e-10
angle = 0.8


# --- math -----------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_figure
    from examples.math.octonions import render

    save_figure(render.draw_fano(core.cycles), "octonions")
    mv, Octonion = core.mv, core.Octonion
    first, second, third = mv.a, mv.b, mv.d
    left_action = core.multiply(first, Octonion)                               # [] Octonion+ <- Octonion+
    generators = core.symmetry_generators(tolerance)                           # [generators] Bivector+
    # Keep the part of an ordinary rotation that belongs to the symmetry space.
    rotation_plane = mv.ab                                                     # [] Bivector+
    symmetry = (generators * generators.reverse().scalar_product(rotation_plane)).sum(axis=0)   # [] Bivector+
    rotations = (stack([rotation_plane, symmetry]) * (angle / 2)).exp()        # [cases] Rotor+
    # Rotate the product, or multiply the rotated factors: their difference.
    defects = (rotations >> core.multiply(first, third)) - core.multiply(rotations >> first, rotations >> third)   # [cases] Octonion+

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
    np.testing.assert_allclose((left_action(left_action) + Octonion).kernel, 0, atol=tolerance)
    np.testing.assert_allclose((core.associator(first, second, third) - 2 * mv.g).kernel, 0, atol=tolerance)
    np.testing.assert_allclose(defects[1].kernel, 0, atol=tolerance)
    assert defects[0].symmetric_conjugate_product() > 0.1


if __name__ == "__main__":
    main()
