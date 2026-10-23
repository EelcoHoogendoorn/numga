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
    rows = list(scenarios.zoo(signatures, 0))
    actions = list(scenarios.centres(tuple(s for s in signatures if sum(s) % 2 == 0)))
    for p, q, _, signs, _ in rows:
        Bivector, _, _, _ = core.signature(p, q)
        inner = (Bivector | Bivector).materialize(core.context)
        np.testing.assert_array_equal(core.invariant_form(Bivector).materialize(core.context).kernel, 2 * (p + q - 2) * inner.kernel)
        values = np.real(signs.to_array())
        assert (values > 0).sum() == p * (p - 1) // 2 + q * (q - 1) // 2 and (values < 0).sum() == p * q
    squares = {(p, q): square for p, q, _, _, square in rows}
    for p, q, action in actions:
        np.testing.assert_allclose(action.to_array() ** 2, squares[p, q].to_array(), atol=1e-12)

    rotor, left_factor, right_factor, along_generator, against_generator = scenarios.split(1)
    np.testing.assert_allclose((along_generator * against_generator - against_generator * along_generator).kernel, 0.0, atol=1e-12)
    np.testing.assert_allclose((left_factor * right_factor - rotor).kernel, 0.0, atol=1e-8)
    for factor in (left_factor, right_factor):
        angles = np.abs(np.angle((factor >> core.Euclidean).eigvals().to_array()))
        np.testing.assert_allclose(angles, angles[0], atol=1e-9)

    azimuths = 6
    left, right, knotted, along, against = scenarios.flows(azimuths, np.array([0.3, 0.9]), 120)
    np.testing.assert_allclose((2 * core.mv.xy + 3 * core.mv.zw - (5 * against - along)).kernel, 0.0, atol=1e-12)
    np.testing.assert_allclose((left[..., 0] - left[..., -1]).kernel, 0.0, atol=1e-5)
    np.testing.assert_allclose((knotted[..., 0] - knotted[..., -1]).kernel, 0.0, atol=1e-5)
    forward = core.linking(left[0, 0], left[-1, azimuths // 3])
    backward = core.linking(right[0, 0], right[-1, azimuths // 3])
    np.testing.assert_allclose(forward.norm().to_array(), 1.0, atol=1e-2)
    np.testing.assert_allclose((forward + backward).kernel, 0.0, atol=1e-2)
    figure = render.draw_flows(left, right, knotted)
    assert isinstance(figure, plt.Figure)
    plt.close(figure)
    assert len(render.animate_flows(left, right, knotted, 3)) == 3
