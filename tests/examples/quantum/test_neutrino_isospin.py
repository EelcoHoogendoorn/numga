"""Tests for neutrino flavor isospin and open quantum system dynamics."""

from __future__ import annotations

import numpy as np

from numga import stack
from examples.quantum.neutrino_isospin import core, scenarios


def test_flavor_probabilities():
    """Verify that pure flavor states and equal mixtures project to exact scalar probabilities."""
    pe_e, pmu_e = core.flavor_probabilities(core.flavor_e)
    np.testing.assert_allclose(float(pe_e.kernel[0]), 1.0, atol=1e-12)
    np.testing.assert_allclose(float(pmu_e.kernel[0]), 0.0, atol=1e-12)

    pe_mu, pmu_mu = core.flavor_probabilities(core.flavor_mu)
    np.testing.assert_allclose(float(pe_mu.kernel[0]), 0.0, atol=1e-12)
    np.testing.assert_allclose(float(pmu_mu.kernel[0]), 1.0, atol=1e-12)


def test_vacuum_oscillation_transfer_map():
    """Verify that extensor transfer maps match the analytical two-flavor formula."""
    theta_deg = 30.0
    theta = np.radians(theta_deg)
    omega = 1.2
    steps = 200

    _, states_iter, _, distances = scenarios.vacuum_oscillations(theta_deg=theta_deg, omega=omega, steps=steps)
    trajectory = stack(list(states_iter), axis=0)

    z_coords = trajectory.cast(core.ga.subspace("z")).kernel.squeeze()
    pe_sim = (1.0 + z_coords) * 0.5
    pe_analytical = 1.0 - np.sin(2.0 * theta) ** 2 * np.sin(0.5 * omega * distances) ** 2

    np.testing.assert_allclose(pe_sim, pe_analytical, atol=1e-5)


def test_wavepacket_decoherence_asymptotic():
    """Verify that Lindblad wavepacket dephasing damps flavor oscillations to cos^4(theta) + sin^4(theta)."""
    theta_deg = 30.0
    theta = np.radians(theta_deg)
    omega = 1.0
    gamma = 0.08
    steps = 400

    _, states_iter, _, _, asymptotic_pe = scenarios.wavepacket_decoherence(
        theta_deg=theta_deg, omega=omega, gamma=gamma, steps=steps
    )
    trajectory = stack(list(states_iter), axis=0)

    z_coords = trajectory.cast(core.ga.subspace("z")).kernel.squeeze()
    pe_sim = (1.0 + z_coords) * 0.5

    # Check asymptotic convergence:
    np.testing.assert_allclose(pe_sim[-1], asymptotic_pe, atol=1e-3)


def test_solar_msw_transfer_channel():
    """Solar MSW effect converts over 95% of core electron neutrinos into muon neutrinos."""
    _, steps_iter, _, _ = scenarios.solar_adiabatic(theta_deg=10.0, omega=1.0, steps=600)
    trajectory = stack([s for s, _ in steps_iter], axis=0)
    z_coords = trajectory.cast(core.ga.subspace("z")).kernel.squeeze()
    pe_arr = (1.0 + z_coords) * 0.5

    np.testing.assert_allclose(pe_arr[0], 1.0, atol=1e-12)
    assert pe_arr[-1] < 0.05


def test_trace_conservation():
    """State scalar trace is strictly conserved under both coherent turning and Lindblad dephasing."""
    theta = np.radians(30.0)
    b_vac = core.vacuum_hamiltonian(1.0, theta)
    rotor_mix = core.mixing_rotor(theta)
    mass_axis = rotor_mix >> core.flavor_mu

    turning = core.coherent_generator(b_vac)
    dephasing = core.dephasing_generator(mass_axis, 0.1)
    gen = turning + dephasing

    step = core.evolution(gen, 0.05)
    s = core.state(core.flavor_e)
    for _ in range(50):
        s = step(s)
        scalar_trace = float(s.select[0].kernel[0])
        np.testing.assert_allclose(scalar_trace, 0.5, atol=1e-12)
