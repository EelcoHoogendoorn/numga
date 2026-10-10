"""Readouts and the noise figure for two qubits kept in four."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.quantum.stabilizer import core

ERROR_NAMES = ("none", "bit flip", "phase flip", "both")
KEPT_COLOUR = "0.6"
ENCODED_COLOUR = "#2980b9"
BARE_COLOUR = "#c0392b"


# --- plumbing -------------------------------------------------------------------------
def scalars(values: core.Scalar) -> np.ndarray:
    """The values of a batch of scalars, for printing."""
    return values.cast(values.algebra.subspace.scalar()).kernel[..., 0]


def print_readings(readings: core.Scalar) -> None:
    """Both checks read after each kind of error on each qubit, `[qubits, kinds, checks]`."""
    values = scalars(readings)
    print("Each entry: the check flipping all four qubits, then the check phase-flipping all four.")
    print(f"{'':10s}" + "".join(f"{name:>14s}" for name in ERROR_NAMES))
    for qubit, row in enumerate(values):
        print(f"qubit {qubit:<4d}" + "".join(f"{f'{flip:+.0f} {phase:+.0f}':>14s}" for flip, phase in row))


def draw_noise(rates: np.ndarray, kept: core.Scalar, encoded_errors: core.Scalar, bare_errors: core.Scalar) -> plt.Figure:
    """How often runs are kept, and how often a kept run, or two bare qubits, come back wrong."""
    figure, ax = plt.subplots(figsize=(6.0, 3.4), layout="constrained")
    ax.plot(rates, scalars(kept), color=KEPT_COLOUR, linestyle="--", label="runs kept")
    ax.plot(rates, scalars(bare_errors), color=BARE_COLOUR, label="two bare qubits wrong")
    ax.plot(rates, scalars(encoded_errors), color=ENCODED_COLOUR, label="kept run wrong")
    ax.set(xlim=(rates[0], rates[-1]), ylim=(0, 1), xlabel="chance of an error on each qubit")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    return figure
