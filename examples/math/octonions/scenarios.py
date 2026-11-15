"""Nonassociative numbers drawn as the Fano plane, their Clifford actions, and fourteen multiplication-preserving
rotations."""

from numga import stack
from examples.math.octonions import core

# The singular values counted as zero, and the angle of the finite turns.
tolerance = 1e-10
angle = 0.8


# --- math -----------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_figure
    from examples.math.octonions import render

    save_figure(render.fano_figure(core.cycles), "octonions")
    mv, Octonion = core.mv, core.Octonion
    first, second, third = mv.a, mv.b, mv.d
    left_action = core.product(first, Octonion)                             # [] Octonion+ <- Octonion+
    generators = core.symmetry_generators(tolerance)                        # [...] Bivector+
    # Keep the part of an ordinary rotation that belongs to the symmetry space.
    rotation_plane = mv.ab                                                  # [] Bivector+
    symmetry = (generators * generators.reverse().scalar_product(rotation_plane)).sum(axis=0)   # [] Bivector+
    rotations = (stack([rotation_plane, symmetry]) * (angle / 2)).exp()     # [2] Rotor+
    # Rotate the product, or multiply the rotated factors: their difference.
    rotated_product = rotations >> core.product(first, third)               # [2] Octonion+
    product_of_rotated = core.product(rotations >> first, rotations >> third)   # [2] Octonion+
    defects = rotated_product - product_of_rotated                          # [2] Octonion+

    print("Left multiplication by a")
    print(left_action.coefficient_table())
    print("(a b) d - a (b d)")
    print(core.associator(first, second, third).coefficient_table())
    print(f"{len(generators)} of {len(core.Bivector.output_subspace)} rotation generators preserve multiplication.")
    print("Ordinary rotation: rotate the product minus multiply the rotated factors")
    print(defects[0].coefficient_table())
    print("G2 rotation: the same difference")
    print(defects[1].coefficient_table())

    # checks
    assert len(generators) == 14
    assert ((left_action(left_action) + Octonion).svdvals() < tolerance).all()
    assert (core.associator(first, second, third) - 2 * mv.g).scalar_norm_squared().abs() < tolerance
    assert defects[1].symmetric_conjugate_product().abs() < tolerance
    assert defects[0].symmetric_conjugate_product() > 0.1


if __name__ == "__main__":
    main()
