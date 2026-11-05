"""The spin groups: in the signature (3, 1) the involution fixes the rotations and negates the boosts,
and the pseudoscalar turns the spinors' coefficients in pairs."""

from __future__ import annotations

import numpy as np

from examples.math.spin_groups import core, render, scenarios


def test_rotations_and_boosts_of_the_signature_3_1():
    """In the signature (3, 1) the involution fixes the three planes of rotations and negates the
    three planes of boosts, and the pseudoscalar has no real eigenvalues on the spinors."""
    Bivector, Even, positives, pseudoscalar = core.signature(3, 1)
    involution = positives >> Bivector
    np.testing.assert_allclose((involution(core.mv.xy) - core.mv.xy).kernel, 0.0, atol=1e-12)
    np.testing.assert_allclose((involution(core.mv.xt) + core.mv.xt).kernel, 0.0, atol=1e-12)
    action = (pseudoscalar * Even).eigvals().to_array()
    np.testing.assert_allclose(np.abs(action.imag), 1.0, atol=1e-12)


def test_one_signature_and_one_frame_of_the_flows_draw():
    list(scenarios.zoo(((3, 1),), scenarios.seed))
    list(scenarios.centres(((3, 1),)))
    left, right, knotted = scenarios.flows(2, scenarios.heights[:1], 2)
    render.draw_flows(left, right, knotted)
    next(render.animate_flows(left, right, knotted, 1))
