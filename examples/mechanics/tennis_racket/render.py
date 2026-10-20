"""Drawing for the tennis racket theorem: rate trajectories and momentum drift."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from numga import Extensor


# --- plumbing -------------------------------------------------------------------------
def draw_trajectories(rates: Extensor, energies: Extensor) -> plt.Figure:
    """The rate's components in every plane over time, one panel per body, labelled by its energy."""
    trajectory = rates.cast(rates.algebra.subspace.bivector()).kernel      # [steps + 1, bodies, planes]
    energy = energies.to_array()[0]                                        # [bodies]
    n_bodies = trajectory.shape[1]

    fig, axes = plt.subplots(n_bodies, 1, figsize=(10, 2.0 * n_bodies), squeeze=False, sharex=True)
    for body, ax in enumerate(axes[:, 0]):
        ax.plot(trajectory[:, body, :])
        ax.set_ylabel(f"E ≈ {int(energy[body])}")
        ax.grid(True, alpha=0.3)
    axes[-1, 0].set_xlabel("Time step")
    fig.tight_layout()
    return fig


def draw_integrator_comparison(curves: list, dt: float) -> plt.Figure:
    """Worst momentum drift over all bodies per step, one panel per dimension."""
    dims = sorted({dimension for dimension, _, _ in curves})
    fig, axes = plt.subplots(1, len(dims), figsize=(5.0 * len(dims), 4.0), squeeze=False, sharey=True)
    for dimension, name, drift in curves:
        ax = axes[0, dims.index(dimension)]
        worst = drift.to_array().max(axis=1)
        ax.semilogy(np.arange(len(worst)) * dt, worst, label=name, linestyle="--" if name == "rk4" else "-")
        ax.set_title(f"{dimension} dimensions")
        ax.set_xlabel("Time")
        ax.grid(True, alpha=0.3)
    axes[0, 0].set_ylabel("World momentum drift")
    axes[0, 0].legend()
    fig.tight_layout()
    return fig
