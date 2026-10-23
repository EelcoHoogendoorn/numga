"""The robot arm's statics and homing, and the tracking animation."""

from __future__ import annotations

from itertools import islice

import numpy as np

from examples.mechanics.robot_arm import render, scenarios


def test_statics_and_homing():
    """The tip force loads the joints by its pairing with each carried axis; homing reaches the target."""
    _, joint_torques = scenarios.statics()
    np.testing.assert_allclose(joint_torques.to_array(), [-2.0, -2.1374, -1.1393], atol=0.01)
    error, _ = scenarios.homing()
    assert error.to_array().max() < 1e-8


def test_tracking_animation_renders():
    frames = list(render.animate_tracking(islice(scenarios.tracking(), 3)))
    assert frames and frames[0].ndim == 3 and all(f.shape == frames[0].shape for f in frames)
