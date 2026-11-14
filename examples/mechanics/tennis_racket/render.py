"""Rate-trajectory and momentum-drift charts for the tennis racket theorem."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from numga import Extensor, NumpyContext

from examples.util import charts


# --- plumbing -------------------------------------------------------------------------
@charts.style()
def trajectories_figure(rates: Extensor, energies: Extensor) -> plt.Figure:
    """Plots the rate components in every coordinate plane (`[..., n_bodies]`) with one panel per body."""
    # Pair each rate against the dual basis bivectors to read out its component in every plane.
    mv = NumpyContext(rates.algebra).multivector
    n_planes = len(rates.algebra.subspace.bivector())
    planes = mv.bivector(np.eye(n_planes))                                              # [n_planes] Bivector+
    trajectory = (rates[..., None] & planes.dual()).to_array()                          # [..., n_planes]
    energy = energies.to_array()[0]                                                     # [...]
    n_bodies = trajectory.shape[1]

    fig, axes = charts.subplots(600, 40 + 100 * n_bodies, nrows=n_bodies, squeeze=False, sharex=True)
    for body_index, ax in enumerate(axes[:, 0]):
        ax.plot(trajectory[:, body_index, :])
        ax.set_ylabel(f"E ≈ {int(energy[body_index])}")
        ax.grid(True, alpha=0.3)
    axes[-1, 0].set_xlabel("Time step")
    fig.tight_layout()
    return fig


@charts.style()
def integrator_comparison_figure(curves: list, dt: float) -> plt.Figure:
    """Plots the worst world-momentum drift across bodies per step, with one panel per dimension."""
    dimensions = sorted({dimension for dimension, _, _ in curves})
    n_dimensions = len(dimensions)
    fig, axes = charts.subplots(charts.DESIGN_WIDTH * n_dimensions // 3, 260, ncols=n_dimensions, squeeze=False, sharey=True)
    for dimension, name, drift in curves:
        ax = axes[0, dimensions.index(dimension)]
        worst = drift.to_array().max(axis=1)
        ax.semilogy(np.arange(len(worst)) * dt, worst, linestyle="--" if name == "rk4" else "-")
        ax.set_xlabel(f"Time ({dimension}D)")
        ax.grid(True, alpha=0.3)
    axes[0, 0].set_ylabel("World momentum drift")
    fig.tight_layout()
    return fig
