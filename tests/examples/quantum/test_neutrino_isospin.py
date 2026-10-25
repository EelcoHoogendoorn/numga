"""Neutrino flavor oscillations: the flavors' probabilities at the poles, the vacuum oscillation against the
two-flavor formula, decoherence settling where it should, the Sun converting electron neutrinos, and the
states' trace kept."""

from __future__ import annotations

import numpy as np

from numga import stack
from examples.quantum.neutrino_isospin import core, scenarios


def electron_probability(trajectory: core.Isospin) -> np.ndarray:
    electron, _ = core.flavor_probabilities(trajectory)
    return electron.to_array()


def test_flavor_probabilities_at_the_poles():
    for flavor, expected in ((core.flavor_e, (1.0, 0.0)), (core.flavor_mu, (0.0, 1.0))):
        electron, muon = core.flavor_probabilities(flavor)
        np.testing.assert_allclose([electron.to_array(), muon.to_array()], expected, atol=1e-12)


def test_vacuum_oscillation_is_the_two_flavor_formula():
    theta_degrees, omega = 30.0, 1.2
    states, _, distances = scenarios.vacuum_oscillations(theta_degrees, omega, 200)
    expected = 1.0 - np.sin(2.0 * np.radians(theta_degrees)) ** 2 * np.sin(0.5 * omega * distances) ** 2
    np.testing.assert_allclose(electron_probability(stack(list(states), axis=0)), expected, atol=1e-5)


def test_decoherence_settles_at_the_mixture():
    states, _, _, settled = scenarios.wavepacket_decoherence(30.0, 1.0, 0.08, 100.0, 400)
    # The tolerance tests the distance followed.
    np.testing.assert_allclose(electron_probability(stack(list(states), axis=0))[-1], settled, atol=1e-3)


def test_the_sun_converts_electron_neutrinos():
    steps, _, _ = scenarios.solar_adiabatic(10.0, 1.0, 100.0, 600)
    electron = electron_probability(stack([state for state, _ in steps], axis=0))
    np.testing.assert_allclose(electron[0], 1.0, atol=1e-12)
    assert electron[-1] < 0.05


def test_the_trace_is_kept():
    theta = np.radians(30.0)
    mass_plane = core.mixing_rotor(theta) >> core.phase_plane
    rates = core.coherent_generator(core.vacuum_hamiltonian(1.0, theta)) + core.dephasing_generator(mass_plane, 0.1)
    step = core.evolution(rates, 0.05)
    rho = core.state(core.flavor_e)
    for _ in range(50):
        rho = step(rho)
        np.testing.assert_allclose((core.one & rho).to_array(), 0.5, atol=1e-12)
