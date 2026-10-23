"""Drawing for the induced dipoles: each cluster in the plane z = 0, its atoms with their dipoles as
arrows, over the dipole an isolated atom would carry in the same field."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

ATOM = np.array([0.15, 0.15, 0.18])
DIPOLE = np.array([0.80, 0.25, 0.20])
ISOLATED = np.array([0.78, 0.78, 0.80])
# The length of the isolated atom's dipole arrow, in units of the plane; every arrow to this scale.
ARROW = 0.5
# The margin around the clusters, in units of the plane.
MARGIN = 0.6


def coordinates(vectors) -> np.ndarray:
    """The x and y coefficients of vectors in the plane z = 0, along the last axis."""
    return vectors.cast(vectors.algebra.subspace("x y")).kernel


def clusters(positions, dipoles, isolated) -> plt.Figure:
    """One panel per cluster, all to one scale: each atom with its dipole as an arrow centred on it,
    over the dipole of an isolated atom in light grey."""
    sites = coordinates(positions)
    arrows = coordinates(dipoles.batch())
    reference = coordinates(isolated)
    scale = ARROW / np.linalg.norm(reference)
    reach = np.abs(sites).max() + MARGIN
    figure, axes = plt.subplots(1, len(sites), figsize=(5 * len(sites), 5))
    for ax, site, arrow in zip(axes, sites, arrows):
        ax.quiver(*site.T, *np.broadcast_to(reference * scale, site.shape).T, color=ISOLATED, pivot="middle",
                  angles="xy", scale_units="xy", scale=1, width=0.016, headwidth=2.5, zorder=1)
        ax.quiver(*site.T, *(arrow * scale).T, color=DIPOLE, pivot="middle",
                  angles="xy", scale_units="xy", scale=1, width=0.008, zorder=2)
        ax.scatter(*site.T, s=10, color=ATOM, zorder=3)
        ax.set(xlim=(-reach, reach), ylim=(-reach, reach), aspect="equal")
        ax.axis("off")
    figure.subplots_adjust(0, 0, 1, 1, wspace=0.02)
    return figure
