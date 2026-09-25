"""The spin groups: the invariant form is the inner product, the involution counts rotations and
boosts, the pseudoscalar splits or complexifies the spinors, and the scenarios pass their checks and
draw."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.math.spin_groups import core, render, scenarios


def test_rotations_and_boosts_of_the_signature_3_1():
    """In the signature (3, 1) the involution fixes the three planes of rotations and negates the
    three planes of boosts, and the pseudoscalar acts on the spinors as the complex unit."""
    Bivector, Even, positives, pseudoscalar = core.signature(3, 1)
    involution = core.involution(Bivector, positives)
    np.testing.assert_allclose((involution(core.mv.xy) - core.mv.xy).kernel, 0.0, atol=1e-12)
    np.testing.assert_allclose((involution(core.mv.xt) + core.mv.xt).kernel, 0.0, atol=1e-12)
    action = (pseudoscalar * Even).eigvals().to_array()
    np.testing.assert_allclose(np.abs(action.imag), 1.0, atol=1e-12)


def test_scenarios_pass_their_checks_and_draw():
    signatures = scenarios.SIGNATURES
    text = render.table(scenarios.zoo(signatures, 0), scenarios.centres(tuple(s for s in signatures if sum(s) % 2 == 0)))
    assert len(text.splitlines()) == len(signatures) + 1
    scenarios.split(1)
    orbits = scenarios.flows(6, np.array([0.3, 0.9]), 120)
    figure = render.draw_flows(*orbits)
    assert isinstance(figure, plt.Figure)
    plt.close(figure)
    assert len(render.animate_flows(*orbits, 3)) == 3
