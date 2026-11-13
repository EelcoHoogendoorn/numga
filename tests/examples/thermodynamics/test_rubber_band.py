"""A rubber band lifting a weight by heat: the band and the weight settled for a step, both runs' first states, the
lamp's and the hand's, and their first frame drawn."""

from examples.thermodynamics.rubber_band import render, scenarios


def test_band_runs_and_draws():
    topology, in_weight, states = scenarios.engine_and_pull(n_settling=1)
    frames = render.animate_together(topology, in_weight, scenarios.lamp, scenarios.room_temperature,
                                     scenarios.lamp_temperature, scenarios.pull_swing, scenarios.pull_drop, states)
    next(frames)
