"""Emission patterns around a coil, in the plane that holds the coil's axis."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from numga import stack
from examples.orientation.wu_experiment.core import Direction, Scalar, Spin, mv

COIL_RADIUS = 0.55
# The coil where the plane of the panels cuts it, a wire at each side: the wire's size in points, and the shares of it
# the cross and the dot take that mark its current going into and coming out of the page.
WIRE_SIZE = 15
CROSS_SHARE, DOT_SHARE = 0.5, 0.3
COIL_COLOUR = "#262b32"


def draw_experiment(spins: Spin, directions: Direction, rates: Scalar) -> plt.Figure:
    """One panel per case `[cases]`: the emission rate along each direction `[cases, directions]` as a
    polar curve in the xz plane, and the coil cut by that plane, with the sense of its current."""
    # The sense of each coil's current, read by pairing its patch with the vertical direction, and each direction
    # read by pairing it with the planes across x and z.
    senses = (spins & mv.z.dual()).to_array()                                     # [cases]
    coordinates = (stack((mv.x, mv.z), axis=-1) & directions[..., None]).to_array()   # [directions, 2]
    curves = rates.to_array()[..., None] * coordinates                            # [cases, directions, 2]
    limit = np.abs(curves).max() * 1.1

    figure, panels = plt.subplots(1, len(senses), figsize=(10, 3.9), dpi=100)
    figure.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.98, wspace=0.05)
    for ax, sense, curve in zip(panels, senses, curves):
        ax.fill(*curve.T, color="#df7246", alpha=0.25, linewidth=0)
        ax.plot(*curve.T, color="#df7246", linewidth=2)
        ax.plot([0, 0], [-limit, limit], color="#aab4bf", linestyle="--", linewidth=0.8)
        # With x to the right and z up, +y runs into the page: a current counterclockwise seen from above goes
        # into the page at +x and comes out of it at -x.
        inward = np.sign(sense) * COIL_RADIUS
        ax.plot([inward, -inward], [0, 0], linestyle="none", marker="o", markersize=WIRE_SIZE, markerfacecolor="white",
                markeredgecolor=COIL_COLOUR, markeredgewidth=2)
        ax.plot(inward, 0, marker="x", markersize=CROSS_SHARE * WIRE_SIZE, markeredgecolor=COIL_COLOUR, markeredgewidth=2)
        ax.plot(-inward, 0, marker="o", markersize=DOT_SHARE * WIRE_SIZE, color=COIL_COLOUR)
        ax.set(xlim=(-limit, limit), ylim=(-limit, limit), aspect="equal")
        ax.set_axis_off()
    return figure
