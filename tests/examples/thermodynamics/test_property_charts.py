"""Tests for the air-conditioning psychrometric chart example."""
import numpy as np

from examples.thermodynamics.property_charts import core, render, scenarios


def test_conditioning_balances_and_renders():
    path, key_states, directions, (cooling, condensate, reheating, supply) = scenarios.conditioning(5)
    inlet, outlet = key_states[0], key_states[1]
    supply_state = path[-1]
    outlet_slope = core.relative_humidity_slope(outlet, scenarios.pressure)

    assert not np.shares_memory(directions.kernel, np.empty(0)) and directions.kernel.size == 4
    np.testing.assert_allclose(((core.mv.t & directions) - 1).kernel, 0, atol=1e-12)
    np.testing.assert_allclose((core.mv.w & directions[0]).kernel, 0, atol=1e-12)
    np.testing.assert_allclose((outlet_slope & directions[1]).kernel, 0, atol=1e-12)
    np.testing.assert_allclose(((core.mv.w & inlet) - (core.mv.w & supply_state) - condensate).kernel, 0, atol=1e-12)
    np.testing.assert_allclose(
        (core.enthalpy(inlet) - core.enthalpy(supply_state) - cooling
         - condensate * core.liquid_heat * scenarios.coil_temperature + reheating).kernel,
        0, atol=1e-10,
    )
    assert np.all(scenarios.relative_humidity_of(path) <= 1 + 1e-10)
    assert supply < scenarios.inlet_relative_humidity

    frame = next(render.conditioning(scenarios.relative_humidity_of, core.enthalpy, path, key_states, directions,
                                     scenarios.air_states(path)))
    assert frame.ndim == 3 and frame.shape[2] == 3
