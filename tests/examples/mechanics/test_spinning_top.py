"""The spinning top renders."""

from __future__ import annotations

import numpy as np

from examples.mechanics.spinning_top import render, scenarios


def test_frame_renders():
    motor, parts, ground = next(scenarios.spin(scenarios.BASE, 0.01, 1e-3, 1))
    frame = render.frame(motor, parts, ground, scenarios.VIEW, scenarios.CENTRE, scenarios.EXTENT, 32)
    assert frame.shape == (32, 32, 3) and frame.dtype == np.uint8
