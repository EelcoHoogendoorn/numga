"""The robot arm scenarios check their kinematics; the tracking animation renders."""

from __future__ import annotations

from itertools import islice

import numpy as np

from examples.mechanics.robot_arm import render, scenarios


def test_statics_and_homing():
    """statics asserts the tip velocity against finite differences; homing asserts the tip reaches the target."""
    _, joint_torques = scenarios.statics()
    np.testing.assert_allclose(joint_torques.to_array(), [-2.0, -2.1374, -1.1393], atol=1e-4)
    scenarios.homing()


def test_tracking_animation_renders():
    frames = render.animate_tracking(islice(scenarios.tracking(), 3))
    assert frames and frames[0].ndim == 3 and all(f.shape == frames[0].shape for f in frames)


def test_mathematics_does_not_import_plotting():
    import subprocess
    import sys

    probe = (
        "import examples.mechanics.robot_arm.core, sys; "
        "print([m for m in sys.modules if m.split('.')[0] in ('matplotlib', 'PIL')])"
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]", out.stdout
