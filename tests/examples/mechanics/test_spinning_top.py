"""The spinning top advances through contact substeps and renders single frames and side-by-side strips."""

from __future__ import annotations

from numga import stack
from examples.mechanics.spinning_top import core, render, scenarios


def test_spinning_top_step_and_render():
    parts, ground, state_stream = scenarios.launch_top(scenarios.base_setup, 2, scenarios.dt, 1)
    first_state = next(state_stream)
    second_state = next(state_stream)
    image = render.frame(first_state[0], parts, ground, scenarios.view, scenarios.view_centre, scenarios.view_extent, 32)
    strip = render.row(stack([first_state[0], second_state[0]]), stack([parts, parts]), ground,
                       scenarios.view, scenarios.view_centre, scenarios.view_extent, 32)
    assert image.shape == (32, 32, 3)
    assert strip.shape == (32, 64, 3)
    assert (core.lean_angle(second_state[0]) < 0.5).all()
