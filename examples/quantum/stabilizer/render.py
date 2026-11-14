"""Tables and charts display check syndromes and post-selected infidelity for the four-qubit stabilizer code."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.quantum.stabilizer import core
from examples.util import charts

error_names = ("none", "bit flip", "phase flip", "both")
kept_colour = "0.6"
encoded_colour = "#2980b9"
bare_colour = "#c0392b"


# --- plumbing -------------------------------------------------------------------------
def syndrome_table(readings: core.Scalar) -> str:
    """Formats both check expectations (`[4, 4, 2] Scalar+`) after each single-qubit error as a text table."""
    values = readings.to_array()
    lines = [
        "Columns list the four-qubit bit-flip check followed by the four-qubit phase-flip check.",
        f"{'':10s}" + "".join(f"{name:>14s}" for name in error_names),
    ]
    for qubit, row in enumerate(values):
        lines.append(f"qubit {qubit:<4d}" + "".join(f"{f'{flip:+.0f} {phase:+.0f}':>14s}" for flip, phase in row))
    return "\n".join(lines)


@charts.style()
def noise_figure(
    rates: np.ndarray,
    kept_fraction: core.Scalar,
    encoded_errors: core.Scalar,
    bare_errors: core.Scalar,
) -> plt.Figure:
    """Plots acceptance probability alongside conditional encoded infidelity and bare-qubit infidelity against the per-qubit error rate."""
    figure, ax = charts.subplots(400, 260, layout="constrained")
    ax.plot(rates, kept_fraction.to_array(), color=kept_colour, linestyle="--")
    ax.plot(rates, bare_errors.to_array(), color=bare_colour)
    ax.plot(rates, encoded_errors.to_array(), color=encoded_colour)
    ax.set(
        xlim=(rates[0], rates[-1]),
        ylim=(0, 1),
        xticks=[rates[0], rates[-1]],
        yticks=[0, 1],
        xlabel="chance of an error on each qubit",
        ylabel="probability",
    )
    ax.spines[["top", "right"]].set_visible(False)
    return figure
