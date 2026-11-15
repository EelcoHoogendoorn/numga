"""Coverage tests for neutrino flavor oscillations (`examples/quantum/neutrino_isospin`)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from numga import stack
from examples.quantum.neutrino_isospin import core, render, scenarios


def test_flavor_probabilities_and_trace():
    for flavor, (expected_e, expected_mu) in ((core.flavor_e, (1.0, 0.0)), (core.flavor_mu, (0.0, 1.0))):
        electron, muon = core.flavor_probabilities(flavor)
        assert ((electron - expected_e).abs() < 1e-12).all()
        assert ((muon - expected_mu).abs() < 1e-12).all()

    theta = np.radians(30.0)
    mixing = (core.mixing_plane * (-theta)).exp()
    mass_plane = mixing >> core.phase_plane
    rates = core.coherent_generator(core.vacuum_hamiltonian(1.0, theta)) + 0.1 * core.relaxation(mass_plane)
    step = core.step_evolution(rates, 0.05)
    density = step(core.density_state(core.flavor_e))
    assert (((core.one & density) - 0.5).abs() < 1e-12).all()


def test_vacuum_decoherence_and_solar_channels():
    states, _, distances = scenarios.vacuum_oscillations(30.0, 1.2, 120)
    vacuum_e, _ = core.flavor_probabilities(stack(list(states)))
    expected = 1.0 - np.sin(2.0 * np.radians(30.0)) ** 2 * np.sin(0.5 * 1.2 * distances) ** 2
    assert ((vacuum_e - expected).abs() < 1e-4).all()

    states, _, _, asymptotic_probability = scenarios.wavepacket_decoherence(30.0, 1.0, 0.08, 100.0, 300)
    decoherence_e, _ = core.flavor_probabilities(stack(list(states)))
    assert ((decoherence_e[-1] - asymptotic_probability).abs() < 2e-3).all()

    steps, _, _, _, _ = scenarios.solar_adiabatic(10.0, 1.0, 100.0, 300)
    solar_e, _ = core.flavor_probabilities(stack([spin for spin, _ in steps]))
    assert (solar_e[0] > 0.99).all() and (solar_e[-1] < 0.05).all()


def test_render_figures_and_animation_frames():
    states, plane, distances = scenarios.vacuum_oscillations(30.0, 1.0, 6)
    history = list(states)
    fig = render.precession_figure(history, plane, distances)
    plt.close(fig)
    frames = list(render.precession_animation(plane, distances, 2, history))
    assert len(frames) == 2 and frames[0].ndim == 3

    dec_states, dec_plane, dec_distances, asymptotic_probability = scenarios.wavepacket_decoherence(
        30.0, 1.0, 0.08, 10.0, 6
    )
    dec_history = list(dec_states)
    fig = render.decoherence_figure(dec_history, dec_plane, dec_distances, asymptotic_probability)
    plt.close(fig)

    solar_steps, start_plane, end_plane, solar_distances, potentials = scenarios.solar_adiabatic(10.0, 1.0, 10.0, 6)
    solar_history = list(solar_steps)
    fig = render.adiabatic_figure(solar_history, solar_distances, potentials)
    plt.close(fig)
    solar_frames = list(render.adiabatic_animation(solar_distances, potentials, start_plane, end_plane, 2, solar_history))
    assert len(solar_frames) == 2 and solar_frames[0].ndim == 3
