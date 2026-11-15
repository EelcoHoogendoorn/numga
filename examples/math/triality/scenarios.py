"""Scene parameters and complete Spin(8) generator decomposition with its common G2 derivations."""

from __future__ import annotations

import numpy as np

from examples.math.triality import core

angle = np.pi / 2
turn_angles = np.array([0, 2 * np.pi, 4 * np.pi])
turn_names = ("Start", "One turn", "Two turns")
action_names = ("Output", "First factor", "Second factor")

# Below this threshold the two multiplication offsets vanish and the generator is a derivation.
tolerance = 1e-10


# --- math -----------------------------------------------------------------------------
def triality() -> tuple[core.Scalar, core.Generator, core.OctonionMap, core.OctonionMap, core.OctonionMap]:
    """The 28 generators (`[28] Generator+`) ordered by squared multiplication offsets (`[28] Scalar+`), and their three actions."""
    left_component, right_component = core.multiplication_components()
    penalties, generators = core.action_modes(left_component, right_component)   # [28] Scalar+, Generator+
    output, left, right = core.linked_actions(generators, left_component, right_component)  # [28] Octonion+ <- Octonion+
    return penalties, generators, output, left, right


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    penalties, generators, output, left, right = triality()
    common_mask = penalties < tolerance
    print(f"triality generators: {len(generators)}, common generators: {common_mask.sum()}")

    # checks
    # Every generator's three actions obey the product rule; fourteen of them are derivations.
    product_rule = (output(core.multiply) - core.multiply(left, core.Octonion)
                    - core.multiply(core.Octonion, right))                       # [28] Octonion+ <- (Octonion+, Octonion+)
    product_rule_defect = product_rule.symmetric_conjugate_product().contract(1, 3).contract(1, 2)
    assert (product_rule_defect.abs() < 1e-8).all()
    assert len(generators) == 28 and common_mask.sum() == 14


if __name__ == "__main__":
    main()
