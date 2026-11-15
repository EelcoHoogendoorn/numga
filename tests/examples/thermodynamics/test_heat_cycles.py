"""Checks that a four-stroke Carnot cycle builds cleanly and renders its first animation frame."""
from examples.thermodynamics.heat_cycles import render, scenarios


def test_cycle_runs_and_draws():
    states, _, _ = scenarios.cycle(1)
    next(render.animate_cycle(scenarios.temperature_of, scenarios.pressure_of, states, scenarios.cycle_states(states)))
