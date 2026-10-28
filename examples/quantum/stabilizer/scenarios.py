"""Two logical qubits stored in four, every single error caught, and the code under noise."""

from __future__ import annotations

import numpy as np

from numga import stack
from examples.quantum.stabilizer import core

# The angles of the two stored logical qubits, and the chances of an error on each qubit.
ANGLES = np.array([0.7, 1.9])
RATES = np.linspace(0.0, 0.5, 51)


# --- math -----------------------------------------------------------------------------
def stored() -> tuple[core.Operator, core.Scalar, core.Scalar]:
    """The stored state, both checks' expectations in it, and those of each logical qubit's flip and phase flip."""
    state = core.encoded(ANGLES)                                               # [] Operator+
    checks = core.n_amplitudes * core.checks.scalar_product(state)             # [checks] Scalar+
    logical = stack([core.logical_flips, core.logical_phase_flips], axis=-1)   # [logical, 2] Operator+
    return state, checks, core.n_amplitudes * logical.scalar_product(state)    # [logical, 2] Scalar+


def single_errors(state: core.Operator) -> core.Scalar:
    """Both checks' expectations after each kind of error on each qubit."""
    corrupted = core.errors * state * core.errors.inverse()                    # [qubits, kinds] Operator+
    return core.n_amplitudes * core.checks.scalar_product(corrupted[..., None])   # [qubits, kinds, checks] Scalar+


def noisy(state: core.Operator) -> tuple[core.Scalar, core.Scalar, core.Scalar]:
    """At each rate: acceptance, conditional infidelity of kept runs, and bare-qubit infidelity."""
    kept, fidelity = core.detected(state, RATES)                               # [rates] Scalar+, Scalar+
    return kept, 1 - fidelity / kept, 1 - core.unprotected(ANGLES, RATES)      # [rates] Scalar+, Scalar+, Scalar+


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_figure
    from examples.quantum.stabilizer import render

    state, checks, logical = stored()
    readings = single_errors(state)
    kept, encoded_errors, bare_errors = noisy(state)
    render.print_readings(readings)
    save_figure(render.draw_noise(RATES, kept, encoded_errors, bare_errors), "stabilizer_noise")

    # checks
    np.testing.assert_allclose(checks.kernel, 1, atol=1e-12)
    np.testing.assert_allclose(logical.kernel[..., 0], np.stack([np.sin(ANGLES), np.cos(ANGLES)], axis=-1), atol=1e-12)
    # Every error but none changes the outcome of at least one check.
    assert (readings.kernel[:, 1:].min(axis=-2) < 0).all()
    # Doubling a small rate quadruples the conditional infidelity, and doubles the bare infidelity.
    np.testing.assert_allclose(encoded_errors.kernel[2] / encoded_errors.kernel[1], 4, rtol=0.05)
    np.testing.assert_allclose(bare_errors.kernel[2] / bare_errors.kernel[1], 2, rtol=0.05)


if __name__ == "__main__":
    main()
