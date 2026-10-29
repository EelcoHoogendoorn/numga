"""Readouts and the noise figure for the four-qubit stabilizer code."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.quantum.stabilizer import core
from examples.util import charts

ERROR_NAMES = ("none", "bit flip", "phase flip", "both")
KEPT_COLOUR = "0.6"
ENCODED_COLOUR = "#2980b9"
BARE_COLOUR = "#c0392b"


# --- plumbing -------------------------------------------------------------------------
def print_readings(readings: core.Scalar) -> None:
    """Both checks read after each kind of error on each qubit, `[qubits, kinds, checks]`."""
    values = readings.to_array()
    print("Each entry: the check flipping all four qubits, then the check phase-flipping all four.")
    print(f"{'':10s}" + "".join(f"{name:>14s}" for name in ERROR_NAMES))
    for qubit, row in enumerate(values):
        print(f"qubit {qubit:<4d}" + "".join(f"{f'{flip:+.0f} {phase:+.0f}':>14s}" for flip, phase in row))


@charts.style()
def draw_noise(rates: np.ndarray, kept: core.Scalar, encoded_errors: core.Scalar, bare_errors: core.Scalar) -> plt.Figure:
    """Acceptance and infidelity, conditional on acceptance for the encoded qubits."""
    # Half the column: three smooth curves and their legend.
    figure, ax = charts.subplots(400, 260, layout="constrained")
    ax.plot(rates, kept.to_array(), color=KEPT_COLOUR, linestyle="--", label="runs kept")
    ax.plot(rates, bare_errors.to_array(), color=BARE_COLOUR, label="bare infidelity")
    ax.plot(rates, encoded_errors.to_array(), color=ENCODED_COLOUR, label="conditional infidelity")
    ax.set(xlim=(rates[0], rates[-1]), ylim=(0, 1), xticks=[], yticks=[], xlabel="chance of an error on each qubit",
           ylabel="probability")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper right", handlelength=1.6)
    return figure
