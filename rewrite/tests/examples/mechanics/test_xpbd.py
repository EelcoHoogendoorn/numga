"""Unit tests for XPBD geometric constraint projection and chain dynamics."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from numga import NumpyContext
from examples.mechanics.xpbd import core, render, scenarios

context = NumpyContext(core.ga)
mv = context.multivector


def test_distance_projection_closes_a_joint():
    """Repeated projection closes the gap between two displaced anchors."""
    chain, partitions = scenarios.chain(context, 2, 0.1, 5e-2, 0.0, 1e-3, 0.5)
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
            pair, joints.anchors, chain.inertia_inv[joints.bodies], joints.compliance, 0.01
        )
    world_anchors = pair >> joints.anchors
    assert (world_anchors[0] & world_anchors[1]).norm().select[0].to_array().max() < 1e-6
    # The fixed link has infinite mass: only link 1 moved.
    np.testing.assert_allclose(pair[0, 0].kernel, motor[0].kernel)


def test_velocity_projection_cancels_relative_anchor_velocity():
    chain, partitions = scenarios.chain(context, 2, 0.1, 5e-2, 1e-9, 1e-3, 0.5)
    joints = partitions[0]
    motors = chain.motor[joints.bodies]
    rates = chain.rate[joints.bodies] + (mv.xw * 0.5 + mv.yz * 0.3) * np.array([[0.0], [1.0]])

    anchor_velocity = joints.anchors & joints.anchors.commutator(core.Rate)

    def relative(rates):
        velocities = motors >> anchor_velocity(rates)
        return (velocities[0] - velocities[1]).norm().select[0].to_array().max()

    assert relative(rates) > 0.01
    resolved = core.project_velocity_constraint(motors, rates, joints.anchors, chain.inertia_inv[joints.bodies])
    assert relative(resolved) < 1e-6


def test_numpy_chain_draws():
    """The scenario asserts closed joints and a fixed first link; the figure draws."""
    assert isinstance(render.draw_chain(*scenarios.swinging_chain(4, 20, 2, 0.02)), plt.Figure)


def test_jax_chain_matches_numpy():
    numpy_centres, _ = scenarios.swinging_chain(4, 5, 2, 0.02)
    jax_centres, jax_gaps = scenarios.swinging_chain_jax(4, 5, 2, 0.02)
    np.testing.assert_allclose(render.euclidean(jax_centres), render.euclidean(numpy_centres), atol=1e-5)
    assert isinstance(render.draw_chain(jax_centres, jax_gaps), plt.Figure)


def test_mathematics_does_not_import_plotting():
    import subprocess
    import sys

    probe = (
        "import examples.mechanics.xpbd.core, sys; "
        "print([m for m in sys.modules if m.split('.')[0] in ('matplotlib', 'PIL')])"
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]", out.stdout
