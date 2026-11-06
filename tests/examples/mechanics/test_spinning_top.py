"""The spinning top takes a frame of one step, and the frame renders."""

from __future__ import annotations

from examples.mechanics.spinning_top import render, scenarios


def test_a_step_of_the_top_renders():
    motor, _, parts, ground = next(scenarios.spin(scenarios.BASE, scenarios.STEP, scenarios.STEP, 1))
    render.frame(motor, parts, ground, scenarios.VIEW, scenarios.centre, scenarios.EXTENT, 32)
