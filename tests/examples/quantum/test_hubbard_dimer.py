"""Hopping exclusion, singlet binding, charge fluctuations and the localized-spin limit."""

from __future__ import annotations

import numpy as np
import pytest

from examples.quantum.hubbard_dimer import core, scenarios


ROUND_OFF_TOLERANCE = 1e-10
STRONG_CHARGE_LIMIT = 0.025
STRONG_SWAP_ERROR = 0.025
EXCHANGE_RELATIVE_ERROR = 0.011


@pytest.fixture(scope="module")
def exchange_history() -> tuple[core.Pair, core.Pair]:
    return scenarios.exchange(scenarios.REPULSIONS, scenarios.PHASE)


def test_hopping_product_rule_exclusion_and_interacting_spin_sectors():
    strength = scenarios.HOPPING
    zero_repulsion = np.zeros(())
    orbitals = core.mv.vector(np.eye(len(core.Orbital.output_subspace)))
    first, second = orbitals[:, None], orbitals[None, :]
    single = core.hopping(strength)
    lifted = core.hamiltonian(strength, zero_repulsion)
    product_rule = (single(first) ^ second) + (first ^ single(second))
    np.testing.assert_allclose(
        (lifted(first ^ second) - product_rule).kernel, 0,
        atol=ROUND_OFF_TOLERANCE, rtol=0,
    )

    # Equal-spin hops into occupied orbitals vanish, and the opposite-spin triplet
    # cancels by exchange signs. Repulsion leaves all three triplets untouched.
    hamiltonians = core.hamiltonian(strength, scenarios.REPULSIONS)
    np.testing.assert_allclose(
        hamiltonians[..., None](core.TRIPLETS).kernel, 0,
        atol=ROUND_OFF_TOLERANCE, rtol=0,
    )
    np.testing.assert_allclose(
        (hamiltonians(core.SINGLET) + 2 * strength * core.SYMMETRIC_DOUBLE).kernel, 0,
        atol=ROUND_OFF_TOLERANCE, rtol=0,
    )
    np.testing.assert_allclose(
        (hamiltonians(core.SYMMETRIC_DOUBLE)
         - scenarios.REPULSIONS * core.SYMMETRIC_DOUBLE
         + 2 * strength * core.SINGLET).kernel, 0,
        atol=ROUND_OFF_TOLERANCE, rtol=0,
    )


def test_spectrum_and_ground_state_match_the_two_site_solution():
    strength, repulsion = scenarios.HOPPING, scenarios.SWEEP
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
    probabilities, doubles, correlation = scenarios.ground_states(repulsion)
    np.testing.assert_allclose(
        energies.kernel[..., 0], expected_energies, atol=ROUND_OFF_TOLERANCE, rtol=0,
    )
    np.testing.assert_allclose(gap, exact_gap, atol=ROUND_OFF_TOLERANCE, rtol=0)
    np.testing.assert_allclose(
        probabilities.kernel[..., 0], expected_probabilities,
        atol=ROUND_OFF_TOLERANCE, rtol=0,
    )
    np.testing.assert_allclose(
        doubles.kernel[..., 0], expected_doubles, atol=ROUND_OFF_TOLERANCE, rtol=0,
    )
    np.testing.assert_allclose(
        correlation.kernel[..., 0], -3 * (1 - expected_doubles) / 4,
        atol=ROUND_OFF_TOLERANCE, rtol=0,
    )


def test_exchange_matches_free_hopping_and_interacting_charge_dynamics(exchange_history):
    strength, repulsion, phase = scenarios.HOPPING, scenarios.REPULSIONS, scenarios.PHASE
    cosine_part, sine_part = exchange_history
    probabilities = (core.probabilities(cosine_part) + core.probabilities(sine_part)).kernel[..., 0]
    zero = np.zeros_like(phase)
    free_probabilities = np.stack([
        np.sin(phase)**2 / 4, zero, np.cos(phase / 2)**4,
        np.sin(phase / 2)**4, zero, np.sin(phase)**2 / 4,
    ], axis=-1)
    np.testing.assert_allclose(
        (cosine_part.scalar_norm_squared() + sine_part.scalar_norm_squared()).kernel, 1,
        atol=ROUND_OFF_TOLERANCE, rtol=0,
    )
    np.testing.assert_allclose(
        probabilities[0], free_probabilities, atol=ROUND_OFF_TOLERANCE, rtol=0,
    )

    # The singlet mixes with the even double occupancy; the triplet stays fixed.
    # Their relative phase determines the probability of exchanging the two spins.
    separation = np.sqrt(repulsion**2 + 16 * strength**2)
    exact_gap = 8 * strength**2 / (separation + repulsion)
    times = phase / exact_gap[:, None]
    charge_phase = separation[:, None] * times / 2
    expected_doubles = 8 * strength**2 / separation[:, None]**2 * np.sin(charge_phase)**2
    singlet_amplitude = np.exp(-1j * repulsion[:, None] * times / 2) * (
        np.cos(charge_phase) + 1j * repulsion[:, None] / separation[:, None] * np.sin(charge_phase)
    )
    expected_swap = np.abs((1 - singlet_amplitude) / 2)**2
    np.testing.assert_allclose(
        (core.expectation(cosine_part, core.DOUBLE_OCCUPANCY)
         + core.expectation(sine_part, core.DOUBLE_OCCUPANCY)).kernel[..., 0], expected_doubles,
        atol=ROUND_OFF_TOLERANCE, rtol=0,
    )
    np.testing.assert_allclose(
        probabilities[..., 3], expected_swap, atol=ROUND_OFF_TOLERANCE, rtol=0,
    )


def test_large_repulsion_approaches_localized_spin_exchange(exchange_history):
    cosine_part, sine_part = exchange_history
    doubles = (core.expectation(cosine_part, core.DOUBLE_OCCUPANCY)
               + core.expectation(sine_part, core.DOUBLE_OCCUPANCY)).kernel[..., 0]
    swapped = (core.probabilities(cosine_part) + core.probabilities(sine_part)).kernel[..., 0][..., 3]
    _, gaps = scenarios.spectrum(scenarios.REPULSIONS)
    perturbative_gap = 4 * scenarios.HOPPING**2 / scenarios.REPULSIONS[-1]
    relative_gap_error = abs(perturbative_gap / gaps[-1] - 1)

    # These bounds test the approximation at the chosen finite repulsion, rather
    # than roundoff: virtual double occupancy leaves small, fast oscillations.
    assert doubles[-1].max() < STRONG_CHARGE_LIMIT
    assert np.max(np.abs(swapped[-1] - scenarios.SPIN_ONLY)) < STRONG_SWAP_ERROR
    assert relative_gap_error < EXCHANGE_RELATIVE_ERROR


def test_the_dials_draw(exchange_history):
    from examples.quantum.hubbard_dimer import render
    parts = [part[:, :3] for part in exchange_history]
    assert len(render.animate_dials(list(scenarios.NAMES), *scenarios.dials(parts))) == 3


def test_a_field_difference_keeps_up_down_and_freezes_the_swap():
    field = core.field_difference()
    np.testing.assert_allclose((field(core.mv.ad) - core.mv.ad).kernel, 0, atol=ROUND_OFF_TOLERANCE, rtol=0)
    cosine_part, sine_part = scenarios.field_exchange(scenarios.FIELD_RATIOS, scenarios.PHASE)
    probabilities = (core.probabilities(cosine_part) + core.probabilities(sine_part)).kernel[..., 0]
    np.testing.assert_allclose(probabilities.sum(axis=-1), 1, atol=ROUND_OFF_TOLERANCE, rtol=0)
    # Without a field difference the swap completes; five times the exchange freezes it.
    assert probabilities[0, :, 3].max() > 0.99
    assert probabilities[-1, :, 3].max() < 0.05


def test_strong_repulsion_holds_one_electron_per_site_until_the_asymmetry_outweighs_it():
    strong = scenarios.REPULSIONS[-1]
    asymmetry = np.array([0.0, 0.5, 2.0]) * strong
    transferred = scenarios.charge_transfer(scenarios.REPULSIONS, asymmetry).kernel[..., 0]   # [cases, samples]
    np.testing.assert_allclose(transferred[:, 0], 0, atol=ROUND_OFF_TOLERANCE, rtol=0)
    # Without repulsion the charge has all but moved over at half the strongest repulsion; with it, it
    # has barely moved, and it has moved over at twice that repulsion.
    assert transferred[0, 1] > 1.9
    assert transferred[-1, 1] < 0.1
    assert transferred[-1, 2] > 1.9
