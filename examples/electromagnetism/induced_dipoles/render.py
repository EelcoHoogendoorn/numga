"""Draws the planar atom clusters with their induced dipole arrows overlaid on the isolated-atom reference dipole."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from numga import stack
from numga.topology import Topology
from numga.render import simplices
from examples.util.panels import offset_panel, overlay

atom_colour = np.array([0.15, 0.15, 0.18])
dipole_colour = np.array([0.80, 0.25, 0.20])
isolated_colour = np.array([0.78, 0.78, 0.80])
# The length of the isolated atom's dipole arrow, in units of the plane; every arrow to this scale.
arrow_length = 0.5
# The margin around the clusters, in units of the plane.
margin = 0.6
# The side of each panel in inches, at the figure's dots per inch.
panel_size, figure_dpi = 5, 48
# The arrows' widths and the lengths of the strokes of the chevrons at their tips, the isolated atom's and the
# dipoles', and the atoms' radius, in pixels.
isolated_width, isolated_chevron = 4.0, 8.0
dipole_width, dipole_chevron = 2.0, 6.0
atom_radius = 2.5


def clusters(positions, dipoles, reference_dipole) -> plt.Figure:
    """Draws one equal-scale panel per cluster (`[n_cases, n_atoms] Direction-`), overlaying each atom's induced dipole (`[n_cases] Dipole-[n_atoms]`) on the isolated-atom reference (`[] Dipole-`)."""
    mv = positions.context.multivector
    n_cases, n_atoms = positions.shape
    # Every dipole to the scale that draws the isolated atom's at the arrow's length.
    scale = arrow_length / reference_dipole.antinorm()                          # [] Scalar+
    arrows = dipoles.batch() * scale                                            # [n_cases, n_atoms] Dipole-
    reference_arrow = reference_dipole * scale                                  # [] Dipole-
    # Each arrow is a chain of two points, its tail and its tip, about its atom, with a chevron at its tip.
    shafts = Topology.chains([2] * n_atoms)
    half_width = float(((stack((mv.x, mv.y)) & positions[..., None]).abs().max() + margin).to_array())
    figure, axes = plt.subplots(1, n_cases, figsize=(panel_size * n_cases, panel_size), dpi=figure_dpi)
    figure.subplots_adjust(0, 0, 1, 1, wspace=0.02)
    for ax, site, arrow in zip(axes, positions, arrows):
        ax.set(xlim=(-half_width, half_width), ylim=(-half_width, half_width), aspect="equal")
        ax.axis("off")
        camera = offset_panel(ax, mv.x, mv.y)
        isolated_ends = stack((site - reference_arrow / 2, site + reference_arrow / 2), axis=-1).reshape(-1)
        dipole_ends = stack((site - arrow / 2, site + arrow / 2), axis=-1).reshape(-1)
        overlay(ax, [
            simplices.strokes(isolated_ends, shafts, isolated_colour, isolated_width, camera,
                              chevrons=np.ones(n_atoms), chevron=isolated_chevron),
            simplices.strokes(dipole_ends, shafts, dipole_colour, dipole_width, camera,
                              chevrons=np.ones(n_atoms), chevron=dipole_chevron),
            simplices.markers(site, atom_colour, atom_radius, camera)])
    return figure
