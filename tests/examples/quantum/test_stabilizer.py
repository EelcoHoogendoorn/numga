"""Two qubits kept in four: the checks, the stored state, single errors and noise."""

import matplotlib.pyplot as plt
import numpy as np

from examples.quantum.stabilizer import core, render, scenarios


def test_checks_commute_with_each_other_and_with_the_logical_qubits():
    np.testing.assert_allclose((core.checks.squared() - 1).kernel, 0, atol=1e-12)
    np.testing.assert_allclose(core.checks[0].commutator(core.checks[1]).kernel, 0, atol=1e-12)
    np.testing.assert_allclose(core.checks[:, None].commutator(core.logical_flips).kernel, 0, atol=1e-12)
    np.testing.assert_allclose(core.checks[:, None].commutator(core.logical_phase_flips).kernel, 0, atol=1e-12)
    # Each logical qubit's flip and phase flip anticommute, and commute with the other qubit's.
    np.testing.assert_allclose(core.logical_flips.anticommutator(core.logical_phase_flips).kernel, 0, atol=1e-12)
    np.testing.assert_allclose(core.logical_flips.commutator(core.logical_phase_flips[::-1]).kernel, 0, atol=1e-12)
    # The four outcomes' projectors split the states completely.
    np.testing.assert_allclose((core.projectors.sum(axis=0) - 1).kernel, 0, atol=1e-12)


def test_stored_state_passes_both_checks_and_every_single_error_is_caught():
    state, checks, logical = scenarios.stored()
    np.testing.assert_allclose(checks.kernel, 1, atol=1e-12)
    np.testing.assert_allclose(logical.kernel[..., 0], np.stack([np.sin(scenarios.ANGLES), np.cos(scenarios.ANGLES)], axis=-1), atol=1e-12)
    readings = scenarios.single_errors(state).kernel[..., 0]                  # [qubits, kinds, checks]
    np.testing.assert_allclose(readings[:, 0], 1, atol=1e-12)
    assert (readings[:, 1:].min(axis=-1) < 0).all()


def test_kept_runs_go_wrong_at_second_order_and_the_figure_draws():
    state, _, _ = scenarios.stored()
    kept, encoded_errors, bare_errors = scenarios.noisy(state)
    np.testing.assert_allclose(encoded_errors.kernel[2] / encoded_errors.kernel[1], 4, rtol=0.05)
    np.testing.assert_allclose(bare_errors.kernel[2] / bare_errors.kernel[1], 2, rtol=0.05)
    assert (kept.kernel <= 1 + 1e-12).all()
    plt.close(render.draw_noise(scenarios.RATES, kept, encoded_errors, bare_errors))
