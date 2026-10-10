"""Drawing for the hanging chain: every Newton iterate of the chain between its pins, over the catenary;
and the largest unbalanced force on a bead at each iterate."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

EARLY = np.array([0.80, 0.80, 0.82])
LATE = np.array([0.15, 0.15, 0.18])
CHAIN = np.array([0.20, 0.40, 0.70])
CATENARY = np.array([0.85, 0.30, 0.20])
PIN = np.array([0.15, 0.15, 0.18])


def coordinates(vectors) -> np.ndarray:
    """The x and y coefficients of vectors in the plane, along the last axis."""
    return np.asarray(vectors.cast(vectors.algebra.subspace("x y")).kernel)


def iterates(chains, residual, catenary, left, right) -> plt.Figure:
    """Two panels. Left: each iterate of the chain as the polyline from pin to pin through its beads,
    earlier ones lighter, the last with its beads marked, over the catenary dashed. Right: the largest
    unbalanced force on a bead at each iterate, in bead weights, on a log scale."""
    beads = coordinates(chains.batch())
    pins = np.broadcast_to(coordinates(left), beads[:, :1].shape), np.broadcast_to(coordinates(right), beads[:, :1].shape)
    lines = np.concatenate([pins[0], beads, pins[1]], axis=1)
    curve = coordinates(catenary)
    largest = np.asarray(residual.batch().to_array()).max(axis=-1)

    figure, (shape, convergence) = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={"width_ratios": [1.3, 1]})
    shades = np.linspace(0, 1, len(lines))[:, None]
    for line, shade in zip(lines[:-1], shades[:-1]):
        shape.plot(*line.T, color=EARLY + shade * (LATE - EARLY), linewidth=1.0)
    shape.plot(*curve.T, color=CATENARY, linewidth=2.5, linestyle="--", label="catenary")
    shape.plot(*lines[-1].T, color=CHAIN, linewidth=1.2, marker="o", markersize=3.5, label="at rest")
    shape.scatter(*np.stack([lines[-1, 0], lines[-1, -1]]).T, s=60, color=PIN, marker="s", zorder=3)
    shape.set(aspect="equal")
    shape.axis("off")
    shape.legend(loc="lower right", frameon=False)

    convergence.semilogy(np.arange(len(largest)), largest, color=CHAIN, marker="o")
    convergence.set(xlabel="Newton step", ylabel="largest unbalanced force, in bead weights")
    convergence.grid(True, which="major", alpha=0.3)
    figure.tight_layout()
    return figure
