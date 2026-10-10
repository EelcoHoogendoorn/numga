"""Two logical qubits stored in four, every single error caught, and the code under noise."""

from __future__ import annotations

import numpy as np

from numga import stack
from examples.quantum.stabilizer import core

# The angles of the two stored logical qubits, and the chances of an error on each qubit.
ANGLES = np.array([0.7, 1.9])
RATES = np.linspace(0.0, 0.5, 51)


# --- math -----------------------------------------------------------------------------
def stored() -> tuple[core.State, core.Scalar, core.Scalar]:
    """The stored state, both checks on it, and each logical qubit's flip and phase flip on it."""
    state = core.encoded(ANGLES)                                               # [] State
    checks = core.expectation(core.checks, state[..., None])                   # [checks] Scalar
    logical = stack([core.logical_flips, core.logical_phase_flips], axis=-1)   # [logical, 2] Full
    return state, checks, core.expectation(logical, state[..., None, None])    # [logical, 2] Scalar


def single_errors(state: core.State) -> core.Scalar:
    """Both checks after each kind of error on each qubit."""
    return core.expectation(core.checks, core.action(core.errors, state)[..., None])  # [qubits, kinds, checks] Scalar


def noisy(state: core.State) -> tuple[core.Scalar, core.Scalar, core.Scalar]:
    """At each rate: acceptance, conditional infidelity of kept runs, and bare-qubit infidelity."""
    kept, fidelity = core.detected(state, RATES)                               # [rates] Scalar each
    return kept, 1 - fidelity / kept, 1 - core.unprotected(ANGLES, RATES)      # [rates] Scalar each


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.animation import save_figure
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
