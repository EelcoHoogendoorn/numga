"""Tests verify stabilizer commutation, single-error syndrome detection, and rendering for the four-qubit code."""

import matplotlib.pyplot as plt
import numpy as np

from examples.quantum.stabilizer import core, render, scenarios


def test_checks_commute_with_each_other_and_with_logical_operators():
    assert ((core.checks.squared() - 1).scalar_norm_squared() < 1e-20).all()
    assert core.checks[0].commutator(core.checks[1]).scalar_norm_squared() < 1e-20
    assert (core.checks[:, None].commutator(core.logical_flips).scalar_norm_squared() < 1e-20).all()
    assert (core.checks[:, None].commutator(core.logical_phase_flips).scalar_norm_squared() < 1e-20).all()
    # Each logical qubit's flip and phase flip anticommute on the same qubit and commute across distinct qubits.
    assert (core.logical_flips.anticommutator(core.logical_phase_flips).scalar_norm_squared() < 1e-20).all()
    assert (core.logical_flips.commutator(core.logical_phase_flips[::-1]).scalar_norm_squared() < 1e-20).all()
    assert (core.projectors.sum(axis=0) - 1).scalar_norm_squared() < 1e-20


def test_encoded_state_passes_checks_and_catches_every_single_error():
    stored_state = core.encoded_state(scenarios.angles)
    check_readings = core.n_amplitudes * core.checks.scalar_product(stored_state)
    logical_readings = core.n_amplitudes * core.logical_operators.scalar_product(stored_state)
    expected_logical = np.stack([np.sin(scenarios.angles), np.cos(scenarios.angles)], axis=-1)
    assert ((check_readings - 1).abs() < 1e-12).all()
    assert ((logical_readings - expected_logical).abs() < 1e-12).all()

    readings = core.single_error_readings(stored_state)
    assert ((readings[:, 0] - 1).abs() < 1e-12).all()
    assert (readings[:, 1:] < 0).any(axis=-1).all()


def test_syndrome_table_and_noise_chart_render():
    stored_state = core.encoded_state(scenarios.angles)
    sample_rates = np.array([0.0, 0.1, 0.2])
    kept_fraction, encoded_errors = core.code_performance(stored_state, sample_rates)
    bare_errors = core.bare_infidelity(scenarios.angles, sample_rates)
    table = render.syndrome_table(core.single_error_readings(stored_state))
    assert "qubit 0" in table
    figure = render.noise_figure(sample_rates, kept_fraction, encoded_errors, bare_errors)
    plt.close(figure)
