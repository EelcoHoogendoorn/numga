"""A Carnot cycle: its states, heat and work at a step a stroke, and its first frame drawn."""
from examples.thermodynamics.heat_cycles import render, scenarios


def test_cycle_runs_and_draws():
    states, _, _ = scenarios.cycle(1)
    next(render.animate_cycle(scenarios.temperature_of, scenarios.pressure_of, states, scenarios.running(states)))
