"""Drawing for the hanging chain: every Newton iterate of the chain between its pins, over the catenary;
and the largest unbalanced force on a bead at each iterate."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from numga import stack

EARLY = np.array([0.80, 0.80, 0.82])
LATE = np.array([0.15, 0.15, 0.18])
CHAIN = np.array([0.20, 0.40, 0.70])
CATENARY = np.array([0.85, 0.30, 0.20])
PIN = np.array([0.15, 0.15, 0.18])


def coordinates(points) -> np.ndarray:
    """The coordinates of points in the plane, along the last axis: their pairings with the lines x = 0 and y = 0."""
    lines = points.context.multivector
    return np.asarray((stack((lines.x, lines.y)) & points[..., None]).to_array())


def iterates(chains, residual, catenary, pins) -> plt.Figure:
    """Two panels. Left: each iterate of the chain as the polyline from pin to pin through its beads,
    earlier ones lighter, the last with its beads marked, over the catenary dashed. Right: the largest
    unbalanced force on a bead at each iterate, in bead weights, on a log scale."""
    beads = coordinates(chains.batch())
    left, right = np.broadcast_to(coordinates(pins.batch())[:, None, None], (2, len(beads), 1, 2))
    lines = np.concatenate([left, beads, right], axis=1)
    curve = coordinates(catenary)
    largest = np.asarray(residual.batch().to_array()).max(axis=-1)

    figure, (shape, convergence) = plt.subplots(1, 2, figsize=(11 / 3, 4.5 / 3), gridspec_kw={"width_ratios": [1.3, 1]})
    shades = np.linspace(0, 1, len(lines))[:, None]
    for line, shade in zip(lines[:-1], shades[:-1]):
        shape.plot(*line.T, color=EARLY + shade * (LATE - EARLY), linewidth=0.4)
    shape.plot(*curve.T, color=CATENARY, linewidth=0.9, linestyle="--")
    shape.plot(*lines[-1].T, color=CHAIN, linewidth=0.5, marker="o", markersize=1.2)
    shape.scatter(*np.stack([lines[-1, 0], lines[-1, -1]]).T, s=6, color=PIN, marker="s", zorder=3)
    shape.set(aspect="equal")
    shape.axis("off")

    convergence.semilogy(np.arange(len(largest)), largest, color=CHAIN, marker="o", markersize=1.5, linewidth=0.6)
    convergence.set_xlabel("Newton step", fontsize=5)
    convergence.set_ylabel("unbalanced force, in weights", fontsize=5)
    convergence.tick_params(labelsize=5, width=0.4, length=2)
    convergence.grid(True, which="major", alpha=0.3)
    figure.tight_layout()
    return figure
