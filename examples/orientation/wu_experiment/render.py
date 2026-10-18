"""Emission patterns around a coil, in the plane that holds the coil's axis."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, FancyArrowPatch

from examples.orientation.wu_experiment.core import Direction, Scalar, Spin, ga

COIL_RADIUS = 0.55
# The coil seen slightly from above: its near half drawn below its centre.
COIL_TILT = 0.22


def draw_experiment(spins: Spin, directions: Direction, rates: Scalar) -> plt.Figure:
    """One panel per case `[cases]`: the emission rate along each direction `[cases, directions]` as a
    polar curve in the xz plane, and the coil with the sense of its current."""
    senses = spins.cast(ga.subspace("xy")).kernel[..., 0]                         # [cases]
    coordinates = directions.cast(ga.subspace("x z")).kernel                       # [directions, 2]
    curves = rates.cast(ga.subspace.scalar()).kernel[..., :1] * coordinates      # [cases, directions, 2]
    limit = np.abs(curves).max() * 1.1

    figure, panels = plt.subplots(1, len(senses), figsize=(10, 3.9), dpi=100)
    figure.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.98, wspace=0.05)
    for ax, sense, curve in zip(panels, senses, curves):
        ax.fill(*curve.T, color="#df7246", alpha=0.25, linewidth=0)
        ax.plot(*curve.T, color="#df7246", linewidth=2)
        ax.plot([0, 0], [-limit, limit], color="#aab4bf", linestyle="--", linewidth=0.8)
        ax.add_patch(Ellipse((0, 0), 2 * COIL_RADIUS, 2 * COIL_RADIUS * COIL_TILT, fill=False,
                             edgecolor="#262b32", linewidth=2))
        # Seen from the side the near half of the loop moves along x for a counterclockwise current.
        heading = np.sign(sense) * 0.25
        ax.add_patch(FancyArrowPatch((-heading, -COIL_RADIUS * COIL_TILT), (heading, -COIL_RADIUS * COIL_TILT),
                                     arrowstyle="-|>", mutation_scale=18, color="#262b32", linewidth=2))
        ax.set(xlim=(-limit, limit), ylim=(-limit, limit), aspect="equal")
        ax.set_axis_off()
    return figure
