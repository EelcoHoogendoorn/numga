"""Two cylinders heated equally, latched and loaded: one slice of heat, and the first frame drawn."""
from examples.thermodynamics.state_space import render, scenarios


def test_heating_runs_and_draws():
    stages = list(scenarios.equal_heating(2.4, 1))
    next(render.heating(scenarios.temperature_of, scenarios.pressure_of, iter(stages)))
