"""A heated fluid under a loaded piston: its heating in a few frames, through each stage, and its first frame drawn."""
from examples.thermodynamics.phase_change import render, scenarios


def test_heating_runs_and_draws():
    temperature = scenarios.boiling_temperature
    states = list(scenarios.heating(temperature, scenarios.coldest, scenarios.hottest, 3))
    next(render.heating(scenarios.pressure_at, scenarios.enthalpy_at, scenarios.coexistence_at, scenarios.isobar_at,
                        temperature, iter(states)))
