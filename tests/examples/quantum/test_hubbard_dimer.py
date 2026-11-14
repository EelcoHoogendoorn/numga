"""Hopping exclusion, singlet binding, charge fluctuations, and the localized-spin limit."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

from examples.quantum.hubbard_dimer import core, render, scenarios
from numga import stack

round_off_tolerance = 1e-10
strong_charge_limit = 0.025
strong_swap_error = 0.025
exchange_relative_error = 0.011


@pytest.fixture(scope="module")
def exchange_history() -> tuple[core.Pair, core.Pair]:
    return scenarios.exchange(scenarios.repulsions, scenarios.phase)


def test_hopping_product_rule_exclusion_and_interacting_spin_sectors():
    strength = scenarios.hopping
    zero_repulsion = np.zeros(())
    first, second = core.orbitals[:, None], core.orbitals[None, :]
    single = -strength * core.hop
    two_electron_hop = core.hamiltonian(strength, zero_repulsion)
    product_rule = (single(first) ^ second) + (first ^ single(second))
    assert ((two_electron_hop(first ^ second) - product_rule).scalar_norm_squared().square_root() < round_off_tolerance).all()

    # Equal-spin hops into occupied orbitals vanish, and the opposite-spin triplet
    # cancels by exchange signs. Repulsion leaves all three triplets untouched.
    hamiltonians = core.hamiltonian(strength, scenarios.repulsions)
    assert (hamiltonians[..., None](core.triplets).scalar_norm_squared().square_root() < round_off_tolerance).all()
    assert ((hamiltonians(core.singlet) + 2 * strength * core.symmetric_double).scalar_norm_squared().square_root() < round_off_tolerance).all()
    assert ((hamiltonians(core.symmetric_double)
             - scenarios.repulsions * core.symmetric_double
             + 2 * strength * core.singlet).scalar_norm_squared().square_root() < round_off_tolerance).all()


def test_spectrum_and_ground_state_match_the_two_site_solution():
    strength, repulsion = scenarios.hopping, scenarios.sweep
    separation = np.sqrt(repulsion**2 + 16 * strength**2)
    exact_gap = 8 * strength**2 / (separation + repulsion)
    zero = np.zeros_like(repulsion)
    expected_energies = np.stack(
        [-exact_gap, zero, zero, zero, repulsion, repulsion + exact_gap], axis=-1,
    )
    expected_doubles = (1 - repulsion / separation) / 2
    expected_probabilities = np.stack([
        expected_doubles / 2, zero, (1 - expected_doubles) / 2,
        (1 - expected_doubles) / 2, zero, expected_doubles / 2,
    ], axis=-1)

    energies, gap = scenarios.spectrum(repulsion)
    probabilities, doubles, alignment = scenarios.ground_states(repulsion)
    assert ((energies - expected_energies).abs() < round_off_tolerance).all()
    assert np.all(np.abs(gap - exact_gap) < round_off_tolerance)
    assert ((probabilities - expected_probabilities).abs() < round_off_tolerance).all()
    assert ((doubles - expected_doubles).abs() < round_off_tolerance).all()
    assert ((alignment - (-3 * (1 - expected_doubles) / 4)).abs() < round_off_tolerance).all()


def test_exchange_matches_free_hopping_and_interacting_charge_dynamics(exchange_history):
    strength, repulsion, phase = scenarios.hopping, scenarios.repulsions, scenarios.phase
    cosine_part, sine_part = exchange_history
    probabilities = core.probabilities(cosine_part) + core.probabilities(sine_part)
    zero = np.zeros_like(phase)
    free_probabilities = np.stack([
        np.sin(phase)**2 / 4, zero, np.cos(phase / 2)**4,
        np.sin(phase / 2)**4, zero, np.sin(phase)**2 / 4,
    ], axis=-1)
    assert ((cosine_part.scalar_norm_squared() + sine_part.scalar_norm_squared() - 1).abs() < round_off_tolerance).all()
    assert ((probabilities[0] - free_probabilities).abs() < round_off_tolerance).all()

    # The singlet mixes with the symmetric double occupancy while the triplet stays fixed;
    # their relative phase governs the spin-swap probability.
    separation = np.sqrt(repulsion**2 + 16 * strength**2)
    exact_gap = 8 * strength**2 / (separation + repulsion)
    times = phase / exact_gap[:, None]
    charge_phase = separation[:, None] * times / 2
    expected_doubles = 8 * strength**2 / separation[:, None]**2 * np.sin(charge_phase)**2
    singlet_amplitude = np.exp(-1j * repulsion[:, None] * times / 2) * (
        np.cos(charge_phase) + 1j * repulsion[:, None] / separation[:, None] * np.sin(charge_phase)
    )
    expected_swap = np.abs((1 - singlet_amplitude) / 2)**2
    total_doubles = core.expectation(cosine_part, core.double_occupancy) + core.expectation(sine_part, core.double_occupancy)
    assert ((total_doubles - expected_doubles).abs() < round_off_tolerance).all()
    assert ((probabilities[..., 3] - expected_swap).abs() < round_off_tolerance).all()


def test_large_repulsion_approaches_localized_spin_exchange(exchange_history):
    cosine_part, sine_part = exchange_history
    doubles = core.expectation(cosine_part, core.double_occupancy) + core.expectation(sine_part, core.double_occupancy)
    swapped = (core.probabilities(cosine_part) + core.probabilities(sine_part))[..., 3]
    _, gaps = scenarios.spectrum(scenarios.repulsions)
    perturbative_gap = 4 * scenarios.hopping**2 / scenarios.repulsions[-1]
    relative_gap_error = abs(perturbative_gap / gaps[-1] - 1)

    assert (doubles[-1] < strong_charge_limit).all()
    assert ((swapped[-1] - scenarios.spin_only).abs() < strong_swap_error).all()
    assert relative_gap_error < exchange_relative_error


def test_a_field_difference_keeps_up_down_and_freezes_the_swap():
    field = core.field_difference
    assert ((field(core.mv.ad) - core.mv.ad).scalar_norm_squared().square_root() < round_off_tolerance).all()
    states = list(scenarios.field_exchange(scenarios.field_ratios, scenarios.phase[::20]))
    cosine_part = stack([state[0] for state in states], axis=1)
    sine_part = stack([state[1] for state in states], axis=1)
    probabilities = core.probabilities(cosine_part) + core.probabilities(sine_part)
    assert ((probabilities.sum(axis=-1) - 1).abs() < round_off_tolerance).all()
    assert bool((probabilities[0, :, 3] > 0.99).any())
    assert (probabilities[-1, :, 3] < 0.05).all()


def test_strong_repulsion_holds_one_electron_per_site_until_the_asymmetry_outweighs_it():
    strong = scenarios.repulsions[-1]
    asymmetry = np.array([0.0, 0.5, 2.0]) * strong
    charge_differences = scenarios.charge_transfer(scenarios.repulsions, asymmetry)
    assert (charge_differences[:, 0].abs() < round_off_tolerance).all()
    assert bool(charge_differences[0, 1] > 1.9)
    assert bool(charge_differences[-1, 1] < 0.1)
    assert bool(charge_differences[-1, 2] > 1.9)


def test_the_scenes_draw_at_a_few_samples():
    hopping, repulsions, phase = scenarios.hopping, scenarios.repulsions, scenarios.phase[:2]
    energies, gap = scenarios.spectrum(scenarios.sweep[:5])
    fig1 = render.spectrum_figure(scenarios.sweep[:5], hopping, energies, gap, scenarios.perturbative[:4], scenarios.strong_repulsion[:4])
    plt.close(fig1)
    probabilities, _, _ = scenarios.ground_states(repulsions)
    fig2 = render.ground_states_figure(scenarios.names, repulsions / hopping, probabilities)
    plt.close(fig2)
    parts = scenarios.exchange(repulsions, phase)
    fig3 = render.exchange_figure(phase, core.probabilities(parts[0]) + core.probabilities(parts[1]), scenarios.spin_only[:2])
    plt.close(fig3)
    dial_gen = render.dials_animation(list(scenarios.names), core.configurations, core.zero_triplet, scenarios.exchange_frames(repulsions, phase))
    next(dial_gen)
    dial_gen.close()
    asymmetries = scenarios.asymmetries[:2]
    fig4 = render.transfer_figure(asymmetries / hopping, scenarios.charge_transfer(repulsions, asymmetries), scenarios.names,
                                  repulsions / hopping)
    plt.close(fig4)
