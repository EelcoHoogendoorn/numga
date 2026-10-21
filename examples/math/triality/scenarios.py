"""Scenes for Spin(8) triality: the 28 generators of rotations of the octonions with their two companions
each, and the 14 among them that all three actions share, the generators of G2."""

from __future__ import annotations

import numpy as np

from examples.math.triality import core

# Below this the two multiplication offsets count as zero: the generator is a derivation.
TOLERANCE = 1e-10


# --- math -----------------------------------------------------------------------------
def triality() -> tuple[core.Scalar, core.Rate, core.OctonionMap, core.OctonionMap, core.OctonionMap]:
    """The generators ordered by their squared multiplication offsets, and each one's three actions."""
    left_component, right_component = core.multiplication_components()
    penalties, rates = core.action_modes(left_component, right_component)   # [generators] Scalar, Rate
    output, left, right = core.linked_actions(rates, left_component, right_component)
    return penalties, rates, output, left, right


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    penalties, rates, output, left, right = triality()
    common = penalties.to_array() < TOLERANCE
    print(f"triality generators: {len(rates)}, common generators: {common.sum()}")

    # checks
    # Every generator's three actions obey the product rule; fourteen of them are derivations.
    product_rule = output(core.multiply) - core.multiply(left, core.Octonion) - core.multiply(core.Octonion, right)
    np.testing.assert_allclose(product_rule.kernel, 0, atol=1e-11)
    assert len(rates) == 28 and common.sum() == 14


if __name__ == "__main__":
    main()
