"""The robot arm's statics and homing, and a frame of its tracking."""

from __future__ import annotations

import numpy as np

from examples.mechanics.robot_arm import render, scenarios


def test_statics_and_homing():
    """The tip force loads the joints by its pairing with each carried axis; homing reaches the target."""
    _, joint_torques = scenarios.statics()
    np.testing.assert_allclose(joint_torques.to_array(), [-2.0, -2.1374, -1.1393], atol=0.01)
    error, _ = scenarios.homing()
    assert error.to_array().max() < 1e-8


def test_a_frame_of_the_tracking_renders():
    list(render.animate_tracking(scenarios.tracking(1)))
