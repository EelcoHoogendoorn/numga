"""Charge readouts and generator connections between the sixteen particle sectors."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from numga import NumpyContext, stack
from numga.algebras import VGA2D
from examples.quantum.particle_charges.core import Scalar
from examples.util.wgsl import simplices
from examples.util.wgsl.camera import flat_panel

background = "#fbfaf7"
particle_colours = np.array([to_rgb(colour) for colour in ("#d6453d", "#3a9a5b", "#2f6fc0", "#d9a227")])
connection_colours = np.array([to_rgb(colour) for colour in ("#4a9ea3", "#8a6fc4", "#db663d")])
# The connections' width and opacity, and the sectors' dots: their radius and, for a hollow one, its hole's, in pixels.
connection_width, connection_opacity = 2.7, 0.82
dot_radius, hole_radius = 6.0, 4.0
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
    offsets = colour_share * (colour_tilt >> (weights[:, 0] * across + weights[:, 1] * up))  # [sectors] Antivector-
    places = centres + offsets                                             # [sectors] Antivector-
    colour_counts = occupations[:, :3].sum(axis=-1)
    occupied_colour = np.argmax(occupations[:, :3], axis=-1)
    missing_colour = np.argmin(occupations[:, :3], axis=-1)
    colour_indices = np.where(colour_counts == 1, occupied_colour, missing_colour)
    colour_indices = np.where((colour_counts == 0) | (colour_counts == 3), 3, colour_indices)
    colours = particle_colours[colour_indices]
    doublets = np.abs(left) > 1e-10
    faces = np.where(doublets[:, None], colours, to_rgb(background))

    figure, axes = plt.subplots(
        1, n_panels, squeeze=False, figsize=(3 * n_panels, 3), facecolor=background,
    )
    # The view reaches past the farthest place across or up, read by pairing with the diagram's planes x and y.
    extent = float((stack((diagram.x, diagram.y)) & places[..., None]).abs().max().to_array()) + 0.35
    for ax in axes[0]:
        ax.set(xlim=(-extent, extent), ylim=(-extent, extent), aspect="equal")
        ax.set_axis_off()
    figure.subplots_adjust(0, 0, 1, 1, wspace=0.035)
    for panel, ax in enumerate(axes[0]):
        response = responses[panel]
        connected = np.stack(np.nonzero(np.triu(response > 1e-10, k=1)), axis=-1)   # [connections, 2]
        camera = flat_panel(ax, diagram.x.algebra)
        simplices.overlay(ax, [
            simplices.strokes(places, simplices.Topology(len(places), connected), connection_colours[panel],
                              connection_width, camera, opacity=connection_opacity),
            # Each sector's dot, hollow where it is no doublet: its rim, under its face.
            simplices.markers(places, colours, dot_radius, camera),
            simplices.markers(places, faces, hole_radius, camera),
        ])
    plt.close(figure)
    return figure
