"""Default logical angles and depolarizing error rates exercise syndrome detection and post-selected fidelity for the four-qubit stabilizer code."""

from __future__ import annotations

import numpy as np

from examples.quantum.stabilizer import core

angles = np.array([0.7, 1.9])                                                  # [2]
n_rates = 51
rates = np.linspace(0.0, 0.5, n_rates)                                         # [n_rates]


def main() -> None:
    """Runs syndrome detection and saves the depolarizing-noise infidelity chart."""
    from examples.quantum.stabilizer import render
    from examples.util.animation import save_figure

    stored_state = core.encoded_state(angles)                                  # [] Operator+
    check_readings = core.n_amplitudes * core.checks.scalar_product(stored_state)  # [2] Scalar+
    logical_readings = core.n_amplitudes * core.logical_operators.scalar_product(stored_state)  # [2, 2] Scalar+
    readings = core.single_error_readings(stored_state)                        # [4, 4, 2] Scalar+
    kept_fraction, encoded_errors = core.code_performance(stored_state, rates)  # [n_rates] Scalar+, Scalar+
    bare_errors = core.bare_infidelity(angles, rates)                          # [n_rates] Scalar+

    print(render.syndrome_table(readings))
    save_figure(render.noise_figure(rates, kept_fraction, encoded_errors, bare_errors), "stabilizer_noise")

    # checks
    assert (core.projectors.sum(axis=0) - 1).scalar_norm_squared() < 1e-20
    assert ((check_readings - 1).abs() < 1e-12).all()
    expected_logical = np.stack([np.sin(angles), np.cos(angles)], axis=-1)
    assert ((logical_readings - expected_logical).abs() < 1e-12).all()
    # Every non-trivial single-qubit error flips at least one check expectation to -1.
    assert ((readings[:, 0] - 1).abs() < 1e-12).all()
    assert (readings[:, 1:] < 0).any(axis=-1).all()
    # Doubling a small error rate quadruples conditional infidelity and doubles bare-qubit infidelity.
    assert (encoded_errors[2] / encoded_errors[1] - 4).abs() < 0.2
    assert (bare_errors[2] / bare_errors[1] - 2).abs() < 0.1


if __name__ == "__main__":
    main()
