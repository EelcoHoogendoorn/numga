"""The hollow top on coarse samples, launched and stepped once, and its setup drawn."""
from __future__ import annotations

from examples.mechanics.tippe_top import core, render, scenarios

# The samples' spacing, coarse enough that the solid takes a handful of them.
coarse_spacing = 0.5


def test_a_launched_top_steps_and_its_setup_draws():
    body = scenarios.body(sample_spacing=coarse_spacing)
    motor, rate = core.launch(body, scenarios.spins, scenarios.lean)
    core.advance(body, motor, rate, scenarios.gravity, scenarios.static_friction, scenarios.dynamic_friction,
                 scenarios.step, 1)
    render.draw_setup(body)
