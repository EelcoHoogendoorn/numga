"""Unit tests for XPBD geometric constraint projection and chain dynamics."""

from __future__ import annotations

import numpy as np
import pytest

from numga import NumpyContext
from examples.mechanics.xpbd import core, render, scenarios

context = NumpyContext(core.ga)
mv = context.multivector


def test_distance_projection_closes_a_joint():
    """Repeated projection closes the gap between two displaced anchors."""
    chain, partitions = scenarios.chain(context, 1, 0.1, scenarios.SEMI_AXES, 0.0, 0.0, 1e-3, 0.5)
    joints = partitions[0]
    assert core.joint_gaps(chain.motor, partitions).to_array().max() < 1e-12

    # Displace link 1 by (2, -1, 3) mm.
    shift = ((mv.xw * 0.001 - mv.yw * 0.0005 + mv.zw * 0.0015) * np.array([0.0, 1.0])).exp()
    motor = chain.motor * shift
    assert core.joint_gaps(motor, partitions).to_array().max() > 0.003

    # The anchor does not respond parallel to the gap, so each projection removes a fraction of it.
    pair = motor[joints.bodies]
    for _ in range(30):
        pair = core.project_distance_constraint(
            pair, joints.anchors, chain.inverse_inertia[joints.bodies], joints.compliance, 0.01
        )
    world_anchors = pair >> joints.anchors
    assert (world_anchors[0] & world_anchors[1]).norm().to_array().max() < 0.0001
    # The pin has infinite mass: only link 1 moved.
    np.testing.assert_allclose(pair[0, 0].kernel, motor[0].kernel)


def test_velocity_projection_cancels_relative_anchor_velocity():
    chain, partitions = scenarios.chain(context, 1, 0.1, scenarios.SEMI_AXES, 0.0, 1e-9, 1e-3, 0.5)
    joints = partitions[0]
    motors = chain.motor[joints.bodies]
    rates = chain.rate[joints.bodies] + (mv.xw * 0.5 + mv.yz * 0.3) * np.array([[0.0], [1.0]])

    anchor_velocity = joints.anchors & joints.anchors.commutator(core.Rate)

    def relative(rates):
        velocities = motors >> anchor_velocity(rates)
        return (velocities[0] - velocities[1]).norm().select[0].to_array().max()

    assert relative(rates) > 0.01
    resolved = core.project_velocity_constraint(motors, rates, joints.anchors, chain.inverse_inertia[joints.bodies])
    assert relative(resolved) < 1e-6


def test_numpy_chain_keeps_its_joints_closed_and_draws():
    """The joints stay closed and the pin fixed; the animation draws."""
    motors, gaps = scenarios.swinging_chain(4, 20, 2, 0.02)
    assert gaps.to_array().max() < 1e-2 * scenarios.LINK_SPACING
    origin = mv.w.dual()
    assert ((motors[:, 0] >> origin) & origin).norm().to_array().max() < 1e-6
    frames = render.animate_chain(motors[:3], scenarios.SEMI_AXES, scenarios.LINK_SPACING)
    assert frames and all(f.shape == frames[0].shape for f in frames)


def test_jax_chain_matches_numpy():
    pytest.importorskip("jax")
    numpy_motors, _ = scenarios.swinging_chain(4, 5, 2, 0.02)
    jax_motors, _ = scenarios.swinging_chain_jax(4, 5, 2, 0.02)
    np.testing.assert_allclose(np.asarray(jax_motors.kernel), numpy_motors.kernel, atol=0.0001)
