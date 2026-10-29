"""Charge readouts and generator connections between the sixteen particle sectors."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from numga import NumpyContext, stack
from numga.algebras import VGA2D
from examples.quantum.particle_charges.core import Scalar

background = "#fbfaf7"
particle_colours = np.array(["#d6453d", "#3a9a5b", "#2f6fc0", "#d9a227"])
connection_colours = ("#4a9ea3", "#8a6fc4", "#db663d")
# The plane of the weight diagrams, its places antivectors, intrinsic: the two weak charges across and up, four
# apart for each unit, and the colour weights about them, turned a little clockwise and drawn in at a share.
diagram = NumpyContext(VGA2D).multivector
across, up = diagram.x.dual(), diagram.y.dual()                            # [] Antivector-, Antivector-
weak_spacing, colour_share = 4, 0.85
colour_tilt = (diagram.xy * (np.radians(12) / 2)).exp()                    # [] Rotor+


# --- plumbing
def draw_connections(
    occupations: np.ndarray,
    left_isospin: Scalar,
    right_isospin: Scalar,
    colour_weights: Scalar,
    strengths: Scalar,
) -> plt.Figure:
    """Weight diagrams whose edges are the computed actions between particle sectors."""
    left = left_isospin.to_array()
    right = right_isospin.to_array()
    weights = colour_weights.to_array()
    responses = strengths.to_array()
    n_panels = responses.shape[0]

    # The two weak charges separate the four multiplets; colour weights separate their states.
    centres = weak_spacing * (left * across + right * up)                  # [sectors] Antivector-
    offsets = colour_share * (colour_tilt >> (weights[:, 0] * across + weights[:, 1] * up))   # [sectors] Antivector-
    # The places read by pairing with the diagram's planes x and y.
    positions = (stack((diagram.x, diagram.y)) & (centres + offsets)[..., None]).to_array()   # [sectors, 2]
    colour_counts = occupations[:, :3].sum(axis=-1)
    occupied_colour = np.argmax(occupations[:, :3], axis=-1)
    missing_colour = np.argmin(occupations[:, :3], axis=-1)
    colour_indices = np.where(colour_counts == 1, occupied_colour, missing_colour)
    colour_indices = np.where((colour_counts == 0) | (colour_counts == 3), 3, colour_indices)
    colours = particle_colours[colour_indices]
    doublets = np.abs(left) > 1e-10
    faces = np.where(doublets, colours, background)

    figure, axes = plt.subplots(
        1, n_panels, squeeze=False, figsize=(3 * n_panels, 3), facecolor=background,
    )
    extent = np.abs(positions).max() + 0.35
    for panel, ax in enumerate(axes[0]):
        response = responses[panel]
        source, target = np.nonzero(np.triu(response > 1e-10, k=1))
        segments = np.stack((positions[source], positions[target]), axis=1)
        widths = 1.95 * np.sqrt(response[source, target] / response.max())
        ax.add_collection(LineCollection(
            segments, colors=connection_colours[panel], linewidths=widths, alpha=0.82, zorder=1,
        ))
        ax.scatter(
            *positions.T, facecolors=faces, edgecolors=colours, s=51, linewidths=1.45, zorder=3,
        )
        ax.set(xlim=(-extent, extent), ylim=(-extent, extent), aspect="equal")
        ax.set_axis_off()
    figure.subplots_adjust(0, 0, 1, 1, wspace=0.035)
    plt.close(figure)
    return figure
