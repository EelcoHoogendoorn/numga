"""Drawing for the tennis racket theorem: rate trajectories and momentum drift."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from numga import Extensor


def draw_trajectories(rates: Extensor, energies: Extensor) -> plt.Figure:
    """Plot the rate components of each body over time, one panel per spin plane."""
    trajectory = rates.cast(rates.algebra.subspace.bivector()).kernel      # [steps, bodies, planes]
    energy = energies.to_array()
    p, planes = rates.algebra.dimension, trajectory.shape[1]

    fig, axes = plt.subplots(planes, 1, figsize=(10, 2.0 * planes), squeeze=False, sharex=True)
    for i, ax in enumerate(axes[:, 0]):
        ax.plot(trajectory[:, i, :])
        ax.set_ylabel(f"E ≈ {int(energy[i])}")
        ax.grid(True, alpha=0.3)
    axes[0, 0].set_title(f"Medial Axis Theorem ({p}D, {planes} Bivector Planes): Angular Velocities")
    axes[-1, 0].set_xlabel("Time step")
    fig.tight_layout()
    return fig


def draw_integrator_comparison(curves: list, dt: float) -> plt.Figure:
    """Worst momentum drift over all bodies per step, one panel per dimension."""
    dims = sorted({p for p, _, _ in curves})
    fig, axes = plt.subplots(1, len(dims), figsize=(5.0 * len(dims), 4.0), squeeze=False, sharey=True)
    for p, name, drift in curves:
        ax = axes[0, dims.index(p)]
        worst = drift.to_array().max(axis=1)
        ax.semilogy(np.arange(len(worst)) * dt, worst, label=name, linestyle="--" if name == "rk4" else "-")
        ax.set_title(f"{p}D, {p * (p - 1) // 2} spin planes, dt = {dt}")
        ax.set_xlabel("Time")
        ax.grid(True, alpha=0.3)
    axes[0, 0].set_ylabel("World momentum drift")
    axes[0, 0].legend()
    fig.tight_layout()
    return fig
