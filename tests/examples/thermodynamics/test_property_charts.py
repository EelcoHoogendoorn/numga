"""A cooling coil preserves dry air, removes water and closes its energy balance."""
import itertools

import numpy as np

from examples.thermodynamics.property_charts import core, render, scenarios


def test_dry_cooling_and_condensation_follow_their_constraints():
    initial = core.air_state(30, np.array([0.2, 0.7]), 101.325)
    path = core.cooling_path(initial[:, None], 12, 101.325, np.linspace(0, 1, 40))
    water = core.humidity(path).to_array()
    relative = core.relative_humidity(path, 101.325).to_array()
    np.testing.assert_allclose(water[0], water[0, 0], atol=1e-12)
    assert np.all(np.diff(water[1]) <= 1e-12)
    assert np.all(relative <= 1 + 1e-10)
    np.testing.assert_allclose(relative[1, -1], 1, atol=1e-10)
    dew = core.dew_point(initial, 101.325)
    dew_states = core.mv.t.dual() * dew + core.mv.w.dual() * core.humidity(initial)
    np.testing.assert_allclose(core.relative_humidity(dew_states, 101.325).to_array(), 1, atol=1e-10)


def test_coil_and_reheater_close_mass_and_energy_balances():
    path, states, directions, (cooling, condensate, reheating, relative) = scenarios.conditioning(32)
    initial, cooled = states
    np.testing.assert_allclose((core.humidity(initial) - core.humidity(path[-1]) - condensate).to_array(), 0, atol=1e-11)
    liquid_heat = condensate * core.LIQUID_HEAT * core.temperature(cooled)
    np.testing.assert_allclose((core.enthalpy(initial) - core.enthalpy(path[-1]) - cooling - liquid_heat + reheating).to_array(),
                               0, atol=1e-10)
    assert 40 < float(cooling.to_array()) < 50
    assert 0.009 < float(condensate.to_array()) < 0.011
    assert 0.5 < float(relative.to_array()) < 0.6
    np.testing.assert_allclose(core.temperature(directions).to_array(), 1, atol=1e-10)
    np.testing.assert_allclose(core.humidity(directions[0]).to_array(), 0, atol=1e-10)
    constraint = core.derivative(scenarios.relative_humidity_of)(cooled)
    np.testing.assert_allclose(constraint(directions[1]).to_array(), 0, atol=1e-10)


def test_conditioning_frames_are_drawn():
    path, states, directions, _ = scenarios.conditioning(6)
    frames = render.conditioning(scenarios.relative_humidity_of, core.enthalpy, path, states, directions,
                                 scenarios.passing(path))
    frame = next(itertools.islice(frames, 4, None))
    assert frame.ndim == 3 and frame.dtype == np.uint8
