"""Coexisting volumes, the mixtures between them, and the isobar a heated fluid walks along."""
import itertools

import numpy as np

from examples.thermodynamics.phase_change import core, render, scenarios

capacity, attraction, exclusion = scenarios.capacity, scenarios.attraction, scenarios.exclusion
round_off = 1e-10


def test_coexistence_has_equal_pressure_and_chemical_potential_on_distinct_branches():
    temperatures = np.linspace(0.6, 0.99, 9)
    coexisting = scenarios.coexistence_at(temperatures)
    np.testing.assert_allclose(core.mismatch(coexisting, temperatures, capacity, attraction, exclusion).kernel, 0,
                               atol=round_off)
    assert np.all(core.liquid_volume(coexisting) < 1)
    assert np.all(core.vapour_volume(coexisting) > 1)


def test_latent_heat_is_temperature_times_entropy_jump_and_obeys_clapeyron():
    temperature, step = 0.85, 1e-4
    coexisting = scenarios.coexistence_at(np.array(temperature))
    liquid, vapour = core.liquid_volume(coexisting), core.vapour_volume(coexisting)
    latent_heat = scenarios.enthalpy_at(vapour, temperature) - scenarios.enthalpy_at(liquid, temperature)
    np.testing.assert_allclose(latent_heat.kernel, (temperature * ((vapour - exclusion) / (liquid - exclusion)).log()).kernel,
                               atol=round_off)
    saturation = lambda warmth: scenarios.pressure_at(core.liquid_volume(scenarios.coexistence_at(np.array(warmth))), warmth)
    rise = (saturation(temperature + step) - saturation(temperature - step)) / (2 * step)
    np.testing.assert_allclose(rise.kernel, (latent_heat / (temperature * (vapour - liquid))).kernel, rtol=1e-6)


def test_heating_walks_the_isobar_in_rising_enthalpy():
    temperature = 0.85
    level = float(scenarios.pressure_at(core.liquid_volume(scenarios.coexistence_at(np.array(temperature))),
                                        temperature).to_array())
    states = list(scenarios.heating(temperature, 0.7, 0.98, 30))
    volumes, temperatures, enthalpies, liquids = (np.array([float(value.to_array()) for value in column])
                                                  for column in zip(*states))
    boiling = temperatures == temperature
    single = ~boiling
    np.testing.assert_allclose(scenarios.pressure_at(core.mv.scalar(volumes[single][:, None]), temperatures[single]).to_array(),
                               level, atol=round_off)
    assert np.all(np.diff(enthalpies) > 0) and np.all(np.diff(volumes) > 0)
    assert liquids[-1] == 0 and np.all(np.diff(liquids[boiling]) < 0)


def test_heating_frames_are_drawn():
    temperature = 0.85
    frames = render.heating(scenarios.pressure_at, scenarios.enthalpy_at, scenarios.coexistence_at, scenarios.isobar_at,
                            temperature, scenarios.heating(temperature, 0.7, 0.98, 12))
    frame = next(itertools.islice(frames, 6, None))
    assert frame.ndim == 3 and frame.dtype == np.uint8
