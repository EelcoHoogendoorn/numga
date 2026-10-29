"""Drawing for the induced dipoles: each cluster in the plane z = 0, its atoms with their dipoles as
arrows, over the dipole an isolated atom would carry in the same field."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from numga import stack

atom_colour = np.array([0.15, 0.15, 0.18])
dipole_colour = np.array([0.80, 0.25, 0.20])
isolated_colour = np.array([0.78, 0.78, 0.80])
# The length of the isolated atom's dipole arrow, in units of the plane; every arrow to this scale.
arrow_length = 0.5
# The margin around the clusters, in units of the plane.
margin = 0.6
# The side of each panel in inches, at the figure's dots per inch.
panel_size, figure_dpi = 5, 48


def coordinates(directions) -> np.ndarray:
    """The x and y coordinates of directions in the plane z = 0, along the last axis `[..., 2]`: each
    direction read by the planes across x and across y."""
    mv = directions.context.multivector
    return (stack((mv.x, mv.y)) & directions[..., None]).to_array()


def clusters(positions, dipoles, isolated) -> plt.Figure:
    """One panel per cluster, all to one scale: each atom with its dipole as an arrow centred on it,
    over the dipole of an isolated atom in light grey."""
    # Every dipole to the scale that draws the isolated atom's at the arrow's length.
    scale = arrow_length / isolated.squared().square_root()
    sites = coordinates(positions)
    arrows = coordinates(dipoles.batch() * scale)
    reference = coordinates(isolated * scale)
    half_width = np.abs(sites).max() + margin
    figure, axes = plt.subplots(1, len(sites), figsize=(panel_size * len(sites), panel_size), dpi=figure_dpi)
    for ax, site, arrow in zip(axes, sites, arrows):
        ax.quiver(*site.T, *np.broadcast_to(reference, site.shape).T, color=isolated_colour, pivot="middle",
                  angles="xy", scale_units="xy", scale=1, width=0.016, headwidth=2.5, zorder=1)
        ax.quiver(*site.T, *arrow.T, color=dipole_colour, pivot="middle",
                  angles="xy", scale_units="xy", scale=1, width=0.008, zorder=2)
        ax.scatter(*site.T, s=10, color=atom_colour, zorder=3)
        ax.set(xlim=(-half_width, half_width), ylim=(-half_width, half_width), aspect="equal")
        ax.axis("off")
    figure.subplots_adjust(0, 0, 1, 1, wspace=0.02)
    return figure
