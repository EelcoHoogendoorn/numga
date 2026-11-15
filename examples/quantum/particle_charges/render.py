"""Weight diagrams of charge readouts and generator connections across particle sectors."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from numga import NumpyContext, stack
from numga.algebras import VGA2D
from examples.quantum.particle_charges.core import Scalar
from numga.topology import Topology
from numga.render import simplices
from examples.util.panels import offset_panel, overlay

background = "#fbfaf7"
particle_colours = np.array([to_rgb(colour) for colour in ("#d6453d", "#3a9a5b", "#2f6fc0", "#d9a227")])
connection_colours = np.array([to_rgb(colour) for colour in ("#4a9ea3", "#8a6fc4", "#db663d")])
connection_width, connection_opacity = 2.7, 0.82
dot_radius, hole_radius = 6.0, 4.0
# The diagram plane represents places as intrinsic antivectors, separating multiplets by weak charges and tilting colour offsets slightly clockwise.
diagram = NumpyContext(VGA2D).multivector
horizontal_offset, vertical_offset = diagram.x.dual(), diagram.y.dual()    # [] Antivector-, Antivector-
weak_spacing, colour_share = 4, 0.85
colour_tilt = (diagram.xy * (np.radians(12) / 2)).exp()                    # [] Rotor+


# --- plumbing
def connections_figure(
    occupations: np.ndarray,
    left_isospin: Scalar,
    right_isospin: Scalar,
    colour_weights: Scalar,
    strengths: Scalar,
) -> plt.Figure:
    """Renders weight diagrams whose edges show nonzero generator connections between particle sectors."""
    left_values = left_isospin.to_array()
    right_values = right_isospin.to_array()
    weight_values = colour_weights.to_array()
    responses = strengths.to_array()
    n_panels = responses.shape[0]
    n_sectors = left_values.shape[0]

    # Weak charges separate the four multiplets while tilted colour weights separate their states.
    centres = weak_spacing * (left_values * horizontal_offset + right_values * vertical_offset)  # [n_sectors] Antivector-
    raw_offsets = weight_values[:, 0] * horizontal_offset + weight_values[:, 1] * vertical_offset  # [n_sectors] Antivector-
    colour_offsets = colour_share * (colour_tilt >> raw_offsets)           # [n_sectors] Antivector-
    places = centres + colour_offsets                                      # [n_sectors] Antivector-
    colour_counts = occupations[:, :3].sum(axis=-1)
    occupied_colour = np.argmax(occupations[:, :3], axis=-1)
    missing_colour = np.argmin(occupations[:, :3], axis=-1)
    colour_indices = np.where(colour_counts == 1, occupied_colour, missing_colour)
    colour_indices = np.where((colour_counts == 0) | (colour_counts == 3), 3, colour_indices)
    colours = particle_colours[colour_indices]
    doublets = np.abs(left_values) > 1e-10
    faces = np.where(doublets[:, None], colours, to_rgb(background))

    figure, axes = plt.subplots(
        1, n_panels, squeeze=False, figsize=(3 * n_panels, 3), facecolor=background,
    )
    # Pairing with the diagram planes reads the farthest horizontal or vertical offset.
    extent = float((stack((diagram.x, diagram.y)) & places[..., None]).abs().max().to_array()) + 0.35
    for ax in axes[0]:
        ax.set(xlim=(-extent, extent), ylim=(-extent, extent), aspect="equal")
        ax.set_axis_off()
    figure.subplots_adjust(0, 0, 1, 1, wspace=0.035)
    for panel_index, ax in enumerate(axes[0]):
        response = responses[panel_index]
        edge_pairs = np.stack(np.nonzero(np.triu(response > 1e-10, k=1)), axis=-1)  # [..., 2]
        camera = offset_panel(ax, diagram.x, diagram.y)
        overlay(ax, [
            simplices.strokes(places, Topology(n_sectors, edge_pairs), connection_colours[panel_index],
                              connection_width, camera, opacity=connection_opacity),
            # Hollow singlet dots show the background face inside the coloured outer rim.
            simplices.markers(places, colours, dot_radius, camera),
            simplices.markers(places, faces, hole_radius, camera),
        ])
    plt.close(figure)
    return figure
