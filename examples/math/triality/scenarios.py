"""This module computes the 28 generators of rotations of the octonions, each with its two companions, and the 14
among them that all three actions share, the generators of G2."""

from __future__ import annotations

import numpy as np

from examples.math.triality import core

# Below this the two multiplication offsets count as zero: the generator is a derivation.
tolerance = 1e-10


# --- math -----------------------------------------------------------------------------
def triality() -> tuple[core.Scalar, core.Generator, core.OctonionMap, core.OctonionMap, core.OctonionMap]:
    """The generators ordered by their squared multiplication offsets, and each one's three actions."""
    left_component, right_component = core.multiplication_components()
    penalties, generators = core.action_modes(left_component, right_component)   # [generators] Scalar+, Generator+
    output, left, right = core.linked_actions(generators, left_component, right_component)
    return penalties, generators, output, left, right


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    penalties, generators, output, left, right = triality()
    common = penalties < tolerance
    print(f"triality generators: {len(generators)}, common generators: {common.sum()}")

    # checks
    # Every generator's three actions obey the product rule; fourteen of them are derivations.
    product_rule = output(core.multiply) - core.multiply(left, core.Octonion) - core.multiply(core.Octonion, right)
    np.testing.assert_allclose(product_rule.kernel, 0, atol=1e-11)
    assert len(generators) == 28 and common.sum() == 14


if __name__ == "__main__":
    main()
