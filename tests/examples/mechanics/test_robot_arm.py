"""Kinematics, static joint torques, homing convergence, and single-frame GPU rendering tests for the robot arm."""

from __future__ import annotations

import numpy as np

from examples.mechanics.robot_arm import core, render, scenarios


def test_statics_and_homing():
    """Rest-pose statics matches the reference joint torques and homing converges onto the loop start."""
    velocity, joint_torques = scenarios.statics()
    assert (core.mv.w & velocity).abs() < 1e-8
    np.testing.assert_allclose(joint_torques.to_array(), [2.0, 2.1374, 1.1393], atol=0.01)
    error, _ = scenarios.homing()
    assert error < 1e-6


def test_tracking_frame_renders():
    """One tracking step renders into a three-channel RGB frame."""
    frames = list(render.tracking_frames(scenarios.tracking(1)))
    assert len(frames) == 1
    assert frames[0].ndim == 3 and frames[0].shape[2] == 3
