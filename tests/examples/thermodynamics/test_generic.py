"""The GENERIC oscillator conserves its energy and never lowers its entropy."""
from __future__ import annotations

import itertools

import numpy as np

from numga import stack
from numga.backend.jax import derivative
from examples.thermodynamics.generic import core, render, scenarios

round_off = 1e-12
parameters = (scenarios.mass, scenarios.stiffness, scenarios.thermal_capacity, scenarios.reference_temperature)


def test_degeneracies_symmetries_and_production():
    mass, stiffness, thermal_capacity, reference_temperature = parameters
    friction = scenarios.friction
    positions, momenta, entropies = np.array([1.5, -2.0, 0.5]), np.array([2.0, -1.0, 0.0]), np.array([0.0, 1.2, 2.5])
    states = core.mv.q.dual() * positions + core.mv.p.dual() * momenta + core.mv.s.dual() * entropies
    energy = core.energy_gradient(states, *parameters)
    mobility = core.friction_mobility(states, mass, friction, thermal_capacity, reference_temperature)
    alpha = core.mv.q * 2.0 + core.mv.p * (-1.5) + core.mv.s * 0.7
    beta = core.mv.q * (-0.5) + core.mv.p * 3.0 + core.mv.s * 1.2
    np.testing.assert_allclose(core.poisson_map(core.entropy_gradient).kernel, 0, atol=round_off)
    np.testing.assert_allclose((core.poisson_map(beta) & alpha).kernel, -(core.poisson_map(alpha) & beta).kernel,
                               atol=round_off)
    np.testing.assert_allclose(mobility(energy).kernel, 0, atol=round_off)
    np.testing.assert_allclose((mobility(beta) & alpha).kernel, (mobility(alpha) & beta).kernel, atol=round_off)
    assert np.all((mobility(alpha) & alpha).kernel >= -round_off)
    reversible, irreversible = core.rates(states, mass, stiffness, friction, thermal_capacity, reference_temperature)
    rate = reversible + irreversible
    np.testing.assert_allclose((energy & rate).kernel, 0, atol=round_off)
    temperatures = reference_temperature * np.exp(entropies / thermal_capacity)
    np.testing.assert_allclose((core.entropy_gradient & rate).to_array(), friction * (momenta / mass) ** 2 / temperatures,
                               atol=round_off)


def test_energy_gradient_is_the_energy_derivative():
    state = core.mv.q.dual() * 0.7 + core.mv.p.dual() * -1.3 + core.mv.s.dual() * 0.4
    reading = core.energy_gradient(state, *parameters) & core.State
    total = lambda state: core.total_energy(state, *parameters)
    np.testing.assert_allclose((reading - derivative(total)(state)).kernel, 0, atol=round_off)


def test_release_conserves_energy_and_heats_the_damper():
    states = stack(list(scenarios.release(150, 2, 0.04)))
    totals = core.total_energy(states, *parameters)
    np.testing.assert_allclose(totals.to_array(), 10, atol=1e-4)
    assert np.all(np.diff((core.mv.s & states).to_array()) >= -1e-10)
    assert float(core.internal_energy(states[-1], scenarios.thermal_capacity, scenarios.reference_temperature).to_array()) > 9.8


def test_damping_frames_are_drawn():
    frames = render.damping(scenarios.energies_of, scenarios.surface_entropy_of, scenarios.rates_of,
                            scenarios.release(6, 2, 0.04), 0.08)
    frame = next(itertools.islice(frames, 3, None))
    assert frame.ndim == 3 and frame.dtype == np.uint8
