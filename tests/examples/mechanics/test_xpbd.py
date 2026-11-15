"""Coverage tests for XPBD distance and velocity constraint projections, NumPy/JAX stepping, and rendering."""

from __future__ import annotations

import numpy as np

from numga import NumpyContext, stack
from examples.mechanics.xpbd import core, render, scenarios

context = NumpyContext(core.ga)
mv = context.multivector


def test_position_projection_closes_displaced_joint():
    motors, _, _, _, inverse_inertias, _, partitions = scenarios.curved_chain(
        context, 1, 0.1, scenarios.semi_axes, 0.0, 0.5,
    )
    pairs, anchors = partitions[0]
    assert (core.joint_separations(motors, partitions) < 1e-8).all()

    shift = ((mv.xw * 0.001 - mv.yw * 0.0005 + mv.zw * 0.0015) * np.array([0, 1])).exp()   # [2] Rotor+
    displaced = motors * shift                                                  # [2] Rotor+
    assert (core.joint_separations(displaced, partitions) > 0.003).all()

    pair_motors = displaced[pairs]                                              # [2, 1] Rotor+
    for _ in range(30):
        pair_motors = core.position_projection(pair_motors, anchors, inverse_inertias[pairs], 0.0, 0.01)
    world_anchors = pair_motors >> anchors                                      # [2, 1] Point-
    assert ((world_anchors[0] & world_anchors[1]).norm() < 1e-4).all()
    np.testing.assert_allclose(pair_motors[0, 0].kernel, displaced[0].kernel)


def test_velocity_projection_cancels_relative_anchor_velocity():
    motors, rates, _, _, inverse_inertias, _, partitions = scenarios.curved_chain(
        context, 1, 0.1, scenarios.semi_axes, 0.0, 0.5,
    )
    pairs, anchors = partitions[0]
    pair_motors = motors[pairs]                                                 # [2, 1] Rotor+
    pair_rates = rates[pairs] + (mv.xw * 0.5 + mv.yz * 0.3) * np.array([[0], [1]])   # [2, 1] Rate+

    anchor_velocity = anchors.regressive(anchors.commutator(core.Rate))

    def relative_speed(current_rates: core.Rate) -> core.Scalar:
        velocities = pair_motors >> anchor_velocity(current_rates)              # [2, 1] Forque-
        return (velocities[1] - velocities[0]).scalar_norm_squared().square_root()

    assert (relative_speed(pair_rates) > 0.01).all()
    resolved = core.velocity_projection(pair_motors, pair_rates, anchors, inverse_inertias[pairs])
    assert (relative_speed(resolved) < 1e-6).all()


def test_one_link_steps_on_numpy_and_jax_and_renders():
    motors, rates, first_moments, inertias, inverse_inertias, gravity, partitions = scenarios.curved_chain(
        context, 1, scenarios.spacing, scenarios.semi_axes, scenarios.curl, scenarios.gravity_acceleration,
    )
    numpy_motors = stack(
        list(
            core.swing_motors(
                motors, rates, first_moments, inertias, inverse_inertias,
                gravity, scenarios.compliance, scenarios.damping, partitions, 1, 1, scenarios.dt,
            )
        )
    )
    jax_motors = stack(list(scenarios.swinging_motors_jax(1, 1, 1, scenarios.dt)))
    np.testing.assert_allclose(numpy_motors.kernel, jax_motors.kernel, atol=1e-6)
    frame = next(render.animate_chain(scenarios.semi_axes, scenarios.spacing, numpy_motors))
    assert frame.dtype == np.uint8 and frame.ndim == 3
