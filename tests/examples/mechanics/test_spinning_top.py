"""The spinning top stays up and keeps spinning over a short run, and renders."""

from __future__ import annotations

import numpy as np

from examples.mechanics.spinning_top import render, scenarios


def test_a_short_spin_stays_up_and_renders():
    states = list(scenarios.spin(scenarios.BASE, 0.1, scenarios.STEP, 20))
    leans = np.array([scenarios.lean(motor).to_array() for motor, _, _, _ in states])
    assert leans.max() < 2 * scenarios.TILT
    motor, _, parts, ground = states[-1]
    frame = render.frame(motor, parts, ground, scenarios.VIEW, scenarios.centre, scenarios.EXTENT, 32)
    assert frame.shape == (32, 32, 3) and frame.dtype == np.uint8
