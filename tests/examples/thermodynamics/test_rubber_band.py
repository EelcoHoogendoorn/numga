"""Settle the hanging rubber band for one step and draw the first side-by-side frame."""

from examples.thermodynamics.rubber_band import render, scenarios


def test_band_runs_and_draws():
    topology, in_weight, states = scenarios.engine_and_pull(1)
    frames = render.animate_together(topology, in_weight, scenarios.lamp, scenarios.room_temperature,
                                     scenarios.lamp_temperature, scenarios.pull_swing, scenarios.pull_drop, states)
    next(frames)
