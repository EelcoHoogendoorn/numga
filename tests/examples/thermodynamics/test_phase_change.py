"""Smoke test stepping isobaric heating across all three regimes and rendering one composite frame."""
from examples.thermodynamics.phase_change import render, scenarios


def test_heating_runs_and_draws():
    temperature = scenarios.boiling_temperature
    states = list(scenarios.heating(temperature, scenarios.cold_temperature, scenarios.hot_temperature, 3))
    next(render.heating(scenarios.pressure_at, scenarios.enthalpy_at, scenarios.coexistence_at, scenarios.isobar_at,
                        temperature, iter(states)))
