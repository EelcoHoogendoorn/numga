"""Heat exchange preserves energy and increases entropy through a positive mobility."""

import itertools

import numpy as np

from numga import stack
from numga.backend.jax import derivative
from examples.thermodynamics.entropy_production import core, render, scenarios

ROUND_OFF = 1e-11


def test_differential_and_mobility_generate_physical_heat_flow():
    capacity_a = 2
    capacity_b = 3
    conductance = 0.7
    temperatures_a = np.array([0.4, 1, 4])
    temperatures_b = np.array([3, 1, 0.6])
    states = core.mv.a.dual() * (capacity_a * temperatures_a) + core.mv.b.dual() * (capacity_b * temperatures_b)
    covector = core.entropy_covector(states, capacity_a, capacity_b)
    differential = derivative(lambda state: core.entropy(state, capacity_a, capacity_b))(states)
    mobility = core.mobility(states, capacity_a, capacity_b, conductance)
    flow = mobility(covector)
    expected_flow = core.transfer * conductance * (temperatures_a - temperatures_b)
    expected_production = conductance * (temperatures_a - temperatures_b) ** 2 / (temperatures_a * temperatures_b)
    probe_a = core.mv.a * np.array([-3, 0, 4]) + core.mv.b * np.array([2, 7, -1])
    probe_b = core.mv.a * np.array([1, -1, 2]) + core.mv.b * np.array([4, 2, 3])

    # checks
    np.testing.assert_allclose(((covector & core.State) - differential).kernel, 0, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose((flow - expected_flow).kernel, 0, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose(mobility(core.energy_covector).kernel, 0, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose(core.total_energy(flow).kernel, 0, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose(((covector & flow) - expected_production).kernel, 0, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose(((probe_b & mobility(probe_a)) - (probe_a & mobility(probe_b))).kernel,
                               0, atol=ROUND_OFF, rtol=0)
    assert np.all((probe_a & mobility(probe_a)).kernel >= -ROUND_OFF)


def test_finite_steps_conserve_energy_increase_entropy_and_reach_equilibrium():
    capacity_a = 2
    capacity_b = 3
    conductance = 0.7
    dt = 0.5
    long_dt = 100
    isolated_conductance = 0
    n_steps = 12
    temperatures_a = np.array([0.4, 1, 4])
    temperatures_b = np.array([3, 1, 0.6])
    initial = core.mv.a.dual() * (capacity_a * temperatures_a) + core.mv.b.dual() * (capacity_b * temperatures_b)
    state = initial
    trajectory = [state]
    for _ in range(n_steps):
        state = core.advance(state, capacity_a, capacity_b, conductance, dt)
        trajectory.append(state)
    states = stack(trajectory)
    entropies = core.entropy(states, capacity_a, capacity_b)
    temperatures = core.temperatures(states, capacity_a, capacity_b)
    decay = np.exp(-conductance * (1 / capacity_a + 1 / capacity_b) * dt * n_steps)
    equilibrium_temperature = core.total_energy(initial) / (capacity_a + capacity_b)
    expected_difference = (temperatures_a - temperatures_b) * decay
    relaxed = core.advance(initial, capacity_a, capacity_b, conductance, long_dt)
    equilibrium = (core.mv.a.dual() * capacity_a + core.mv.b.dual() * capacity_b) * equilibrium_temperature
    no_transfer = core.advance(initial, capacity_a, capacity_b, isolated_conductance, long_dt)

    # checks
    np.testing.assert_allclose((core.total_energy(states) - core.total_energy(initial)).kernel,
                               0, atol=ROUND_OFF, rtol=0)
    assert np.all((entropies[1:] - entropies[:-1]).kernel >= -ROUND_OFF)
    assert np.all(core.energy_a(states).kernel > 0)
    assert np.all(core.energy_b(states).kernel > 0)
    np.testing.assert_allclose((temperatures[0][-1] - temperatures[1][-1] - expected_difference).kernel,
                               0, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose((relaxed - equilibrium).kernel, 0, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose((no_transfer - initial).kernel, 0, atol=ROUND_OFF, rtol=0)


def test_exchange_frames_are_drawn():
    frames = render.exchange(scenarios.entropy_of, scenarios.temperatures_of, core.transfer,
                             np.array([scenarios.CAPACITY_A, scenarios.CAPACITY_B]), scenarios.exchange(6, 5))
    frame = next(itertools.islice(frames, 3, None))
    assert frame.ndim == 3 and frame.dtype == np.uint8
