"""Drawing for the tennis racket theorem: rate trajectories and momentum drift."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from numga import Extensor, NumpyContext

from examples.util import charts


# --- plumbing -------------------------------------------------------------------------
@charts.style()
def draw_trajectories(rates: Extensor, energies: Extensor) -> plt.Figure:
    """The rate's components in every plane over time, one panel per body, labelled by its energy."""
    # The basis planes, and the rate's part in each: its pairing with the plane's dual.
    mv = NumpyContext(rates.algebra).multivector
    planes = mv.bivector(np.eye(len(rates.algebra.subspace.bivector())))
    trajectory = (planes.dual() & rates[..., None]).to_array()            # [steps + 1, bodies, planes]
    energy = energies.to_array()[0]                                        # [bodies]
    n_bodies = trajectory.shape[1]

    # Three quarters of the column, for the flips to stand apart along the run; a short strip per body, room enough
    # for its three ticks and its energy.
    fig, axes = charts.subplots(600, 40 + 100 * n_bodies, nrows=n_bodies, squeeze=False, sharex=True)
    for body, ax in enumerate(axes[:, 0]):
        ax.plot(trajectory[:, body, :])
        ax.set_ylabel(f"E ≈ {int(energy[body])}")
        ax.grid(True, alpha=0.3)
    axes[-1, 0].set_xlabel("Time step")
    fig.tight_layout()
    return fig


@charts.style()
def draw_integrator_comparison(curves: list, dt: float) -> plt.Figure:
    """Worst momentum drift over all bodies per step, one panel per dimension."""
    dims = sorted({dimension for dimension, _, _ in curves})
    # A third of the column a dimension, for the drift's wiggles over ten decades.
    fig, axes = charts.subplots(charts.DESIGN_WIDTH * len(dims) // 3, 260, ncols=len(dims), squeeze=False, sharey=True)
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
