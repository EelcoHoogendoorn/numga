"""The first law and the equal circulations of heat and work, and a Carnot cycle's exchanges."""
import itertools

import numpy as np

from examples.thermodynamics.heat_cycles import core, render, scenarios

ROUND_OFF = 1e-12


def test_first_law_and_equal_circulations():
    capacity = scenarios.CAPACITY
    states = core.mv.s.dual() * np.array([0, 0.5, 1]) + core.mv.v.dual() * np.array([0.6, 1.2, 2.5])
    differential = core.derivative(lambda state: core.energy(state, capacity))(states)
    temperature, pressure = differential(core.mv.s.dual()), -differential(core.mv.v.dual())
    heat_area, work_area = core.circulations(states, capacity)
    np.testing.assert_allclose((pressure * core.volume(states) - temperature).kernel, 0, atol=ROUND_OFF)
    np.testing.assert_allclose((temperature * core.entropy - pressure * core.volume - differential).kernel, 0, atol=ROUND_OFF)
    np.testing.assert_allclose((heat_area - work_area).kernel, 0, atol=ROUND_OFF)
    np.testing.assert_allclose(heat_area(core.mv.s.dual(), core.mv.v.dual()).kernel,
                               (temperature / (capacity * core.volume(states))).kernel, atol=ROUND_OFF)


def test_carnot_cycle_closes_and_turns_a_third_of_its_heat_into_work():
    n_steps = 16
    states, heat, work = scenarios.cycle(n_steps)
    energy = core.energy(states, scenarios.CAPACITY)
    np.testing.assert_allclose((states[-1] - states[0]).kernel, 0, atol=ROUND_OFF)
    np.testing.assert_allclose((heat - work - (energy - energy[0])).kernel, 0, atol=1e-10)
    np.testing.assert_allclose(scenarios.temperature_of(states[:n_steps + 1]).to_array(), scenarios.HOT, atol=ROUND_OFF)
    np.testing.assert_allclose(core.entropy(states[n_steps:2 * n_steps + 1]).to_array(), scenarios.ENTROPY_SPAN,
                               atol=ROUND_OFF)
    np.testing.assert_allclose(work[-1].to_array(), (scenarios.HOT - scenarios.COLD) * scenarios.ENTROPY_SPAN, atol=1e-10)
    np.testing.assert_allclose((work[-1] / heat[n_steps]).to_array(), 1 - scenarios.COLD / scenarios.HOT, atol=1e-10)


def test_cycle_frames_are_drawn():
    states, _, _ = scenarios.cycle(4)
    frames = render.animate_cycle(scenarios.temperature_of, scenarios.pressure_of, states, scenarios.running(states))
    frame = next(itertools.islice(frames, 5, None))
    assert frame.ndim == 3 and frame.dtype == np.uint8
