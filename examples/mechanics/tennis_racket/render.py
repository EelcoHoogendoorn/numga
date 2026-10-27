"""Drawing for the tennis racket theorem: rate trajectories and momentum drift."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from numga import Extensor, NumpyContext

# The size of the text and the width of the lines, for pictures a third of matplotlib's default size.
FONT = 6
LINE = 0.8


# --- plumbing -------------------------------------------------------------------------
def draw_trajectories(rates: Extensor, energies: Extensor) -> plt.Figure:
    """The rate's components in every plane over time, one panel per body, labelled by its energy."""
    # The basis planes, and the rate's part in each: its pairing with the plane's dual.
    mv = NumpyContext(rates.algebra).multivector
    planes = mv.bivector(np.eye(len(rates.algebra.subspace.bivector())))
    trajectory = (planes.dual() & rates[..., None]).to_array()            # [steps + 1, bodies, planes]
    energy = energies.to_array()[0]                                        # [bodies]
    n_bodies = trajectory.shape[1]

    fig, axes = plt.subplots(n_bodies, 1, figsize=(10 / 3, 2.0 * n_bodies / 3), squeeze=False, sharex=True)
    for body, ax in enumerate(axes[:, 0]):
        ax.plot(trajectory[:, body, :], linewidth=LINE)
        ax.set_ylabel(f"E ≈ {int(energy[body])}", fontsize=FONT)
        ax.tick_params(labelsize=FONT, width=LINE / 2, length=2)
        ax.grid(True, alpha=0.3, linewidth=LINE / 2)
    axes[-1, 0].set_xlabel("Time step", fontsize=FONT)
    fig.tight_layout()
    return fig


def draw_integrator_comparison(curves: list, dt: float) -> plt.Figure:
    """Worst momentum drift over all bodies per step, one panel per dimension."""
    dims = sorted({dimension for dimension, _, _ in curves})
    fig, axes = plt.subplots(1, len(dims), figsize=(5.0 * len(dims) / 3, 4.0 / 3 + 0.3), squeeze=False, sharey=True)
    for dimension, name, drift in curves:
        ax = axes[0, dims.index(dimension)]
        worst = drift.to_array().max(axis=1)
        ax.semilogy(np.arange(len(worst)) * dt, worst, label=name, linestyle="--" if name == "rk4" else "-", linewidth=LINE)
        ax.set_title(f"{dimension} dimensions", fontsize=FONT)
        ax.set_xlabel("Time", fontsize=FONT)
        ax.tick_params(labelsize=FONT, width=LINE / 2, length=2)
        ax.grid(True, alpha=0.3, linewidth=LINE / 2)
    axes[0, 0].set_ylabel("World momentum drift", fontsize=FONT)
    axes[0, 0].legend(fontsize=FONT)
    fig.tight_layout()
    return fig
