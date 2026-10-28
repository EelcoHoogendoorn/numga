"""The equation of state, constrained heat capacities, and the energy balance of two heated cylinders."""
import itertools
from functools import partial

import numpy as np

from numga import stack
from examples.thermodynamics.state_space import core, render, scenarios

round_off = 1e-12
thermal_capacity = scenarios.thermal_capacity


def test_energy_derivatives_and_constrained_heat_capacities():
    entropies = np.array([-0.3, 0.1, 0.8])
    volumes = np.array([0.7, 1.1, 1.8])
    states = core.mv.s.dual() * entropies + core.mv.v.dual() * volumes
    potential = partial(core.energy, thermal_capacity=thermal_capacity)
    temperature, pressure, temperature_change, pressure_change = core.response(states, thermal_capacity)
    np.testing.assert_allclose((temperature - potential(states) / thermal_capacity).kernel, 0, atol=round_off)
    np.testing.assert_allclose((pressure * core.volume(states) - temperature).kernel, 0, atol=round_off)
    # A Maxwell relation: the temperature's change with volume is minus the pressure's with entropy.
    np.testing.assert_allclose((temperature_change(core.mv.v.dual()) + pressure_change(core.mv.s.dual())).kernel, 0,
                               atol=round_off)
    paired = states[:, None].broadcast_to((3, 2))
    constraints = stack((core.volume.broadcast_to((3,)), pressure_change), axis=-1)
    directions, capacities = core.heating_direction(paired, constraints, thermal_capacity)
    np.testing.assert_allclose((temperature_change[:, None](directions) - 1).kernel, 0, atol=round_off)
    np.testing.assert_allclose(constraints(directions).kernel, 0, atol=round_off)
    np.testing.assert_allclose(capacities.to_array(), np.broadcast_to([thermal_capacity, thermal_capacity + 1], (3, 2)),
                               atol=round_off)


def test_equal_heating_conserves_energy_under_both_constraints():
    total_heat = 2.4
    stages = list(scenarios.equal_heating(total_heat, 48))
    start, end = stages[0][0], stages[-1][0]
    work = scenarios.pressure_of(start) * (core.volume(end) - core.volume(start))
    energy_change = core.energy(end, thermal_capacity) - core.energy(start, thermal_capacity)
    np.testing.assert_allclose((energy_change + work).to_array(), total_heat, rtol=1e-3)
    np.testing.assert_allclose(core.volume(end[0]).to_array(), 1, atol=round_off)
    np.testing.assert_allclose(scenarios.pressure_of(end[1]).to_array(), 1, rtol=1e-3)
    np.testing.assert_allclose(scenarios.temperature_of(end).to_array(), [2.6, 1.96], rtol=1e-3)


def test_heating_frames_are_drawn():
    frames = render.heating(scenarios.temperature_of, scenarios.pressure_of, scenarios.equal_heating(2.4, 8))
    frame = next(itertools.islice(frames, 4, None))
    assert frame.ndim == 3 and frame.dtype == np.uint8
