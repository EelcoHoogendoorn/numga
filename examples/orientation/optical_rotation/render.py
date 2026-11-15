"""Perspective panels unfold outgoing and returning polarizations along a straight beam path."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from numga import concatenate
from examples.orientation.optical_rotation.core import Direction, Field, mv
from numga.render import solids
from examples.util.panels import show

# Spatial slice of the medium rest frame, extracting spatial offsets from spacetime antivectors.
space_slice = mv.t ^ (Direction & -mv.t.dual())                                   # [] Direction- <- Direction-

sm = solids.mv
leg_colours = np.array([to_rgb("#df7246"), to_rgb("#238c9b")])
incident_colour, beam_colour = np.array(to_rgb("#262b32")), np.array(to_rgb("#75808a"))
reflector_colour = np.array(to_rgb("#dfe5ec"))
# Drawn half-length of each polarization segment in mm.
line_half_width = 0.45
# Incident-polarization and reflector reaches in polarization half-lengths, dash count and fill fraction, and plate style.
incident_reach, reflector_reach = 1.3, 1.4
n_dashes, dash_fill = 7, 0.6
reflector_thickness, reflector_opacity = 0.03, 0.3
# Stroke widths of polarization lines, incident reference dashes, and the central beam axis in pixels.
line_width, dash_width, beam_width = 2.2, 1.1, 0.85
# Camera elevation and azimuth in radians, viewport margin fraction, and pixel width per panel.
elevation, azimuth = np.radians(12.0), np.radians(35.0)
margin, n_pixels = 0.04, 300
# Half-turn about `x + z` mapping the beam axis `z` horizontal along `x` and the initial polarization `x` vertical along `z`.
upright_turn = ((sm.yz + sm.xy) * (np.pi / (2 * np.sqrt(2)))).exp()               # [] Rotor+


# --- plumbing -------------------------------------------------------------------------
def isolator_figure(fields: Field, depths: np.ndarray) -> plt.Figure:
    """Unfolded outgoing and returning electric `fields` (`[n_media, n_setups, 2, n_depths] Field+`)
    sampled at `depths` (`[n_depths]`) render as one 3D polarization panel per medium and setup."""
    # Intersecting each field's dual plane with the rest-frame time 1-form recovers its spatial polarization direction.
    polarizations = fields.dual_inverse() ^ mv.t                                  # [n_media, n_setups, 2, n_depths] Direction-
    beam_axis = mv.z.dual()                                                       # [] Direction-
    path_length = depths[-1]
    beam_places = np.stack([depths, path_length + depths]) * beam_axis            # [2, n_depths] Direction-
    end_places = np.array([0.0, 2 * path_length]) * beam_axis                     # [2] Direction-
    n_media, n_setups = fields.shape[:2]

    # Far-end reflector plate and bounding corners spanning both unfolded legs.
    reflector_half_width = reflector_reach * line_half_width
    reflector_square = (reflector_half_width * (mv.x.dual() * np.array([-1, 1, 1, -1])
                                                + mv.y.dual() * np.array([-1, -1, 1, 1]))
                        + path_length * beam_axis)                                # [4] Direction-
    box_corners = concatenate([reflector_square - path_length * beam_axis,
                               reflector_square + path_length * beam_axis])       # [8] Direction-
    # Embed rest-frame places around the scene origin and polarization directions as points at infinity.
    scene_corners = solids.origin + (upright_turn >> solids.context.include_from_algebra(space_slice(box_corners)))
    scene_square = solids.origin + (upright_turn >> solids.context.include_from_algebra(space_slice(reflector_square)))
    beam_ends = solids.origin + (upright_turn >> solids.context.include_from_algebra(space_slice(end_places)))
    stations = solids.origin + (upright_turn >> solids.context.include_from_algebra(space_slice(beam_places)))
    directions = upright_turn >> solids.context.include_from_algebra(polarizations)
    view = solids.view([scene_corners], 0.0, elevation, azimuth, margin, n_pixels)
    reflector = solids.plates(scene_square[None], reflector_thickness)
    beam_rod = solids.rods(beam_ends[0], beam_ends[1], beam_width / 2 * view.pixel)[None]
    # Dash endpoints as fractions of the incident reference segment's half-length.
    dash_edges = np.linspace(-1, 1, n_dashes + 1)
    dash_gap = (1 - dash_fill) * (dash_edges[1] - dash_edges[0]) / 2
    dash_starts, dash_ends = dash_edges[:-1] + dash_gap, dash_edges[1:] - dash_gap  # [n_dashes]

    n_rows, n_columns = view.shape
    figure = plt.figure(figsize=(n_media * n_columns / 100, n_setups * n_rows / 100), dpi=100)
    for medium in range(n_media):
        for setup in range(n_setups):
            half_lines = line_half_width * directions[medium, setup]              # [2, n_depths] Point-
            lines = solids.rods(stations - half_lines, stations + half_lines,
                                line_width / 2 * view.pixel)                      # [2, n_depths, 3]
            incident_half_line = incident_reach * line_half_width * directions[medium, setup, 0, 0]  # [] Point-
            dashes = solids.rods(beam_ends[:, None] + dash_starts * incident_half_line,
                                 beam_ends[:, None] + dash_ends * incident_half_line,
                                 dash_width / 2 * view.pixel)                     # [2, n_dashes, 3]
            parts = [(beam_rod, beam_colour), (dashes.reshape(-1, 3), incident_colour),
                     *((lines[leg], colour) for leg, colour in enumerate(leg_colours))]
            image = solids.draw(parts, view, np.ones(3), inks=[],
                                translucent=[(reflector, reflector_colour, reflector_opacity)], meshes=[])
            ax = figure.add_axes((medium / n_media, 1 - (setup + 1) / n_setups, 1 / n_media, 1 / n_setups))
            show(ax, image, view)
    return figure
