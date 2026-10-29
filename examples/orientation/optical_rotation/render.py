"""Polarizations along a beam, out to a reflector and back, unfolded into one straight path."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from numga import stack
from examples.orientation.optical_rotation.core import Field
from examples.util.wgsl import show, solids

sm = solids.mv
leg_colours = np.array([to_rgb("#df7246"), to_rgb("#238c9b")])
entering_colour, beam_colour = np.array(to_rgb("#262b32")), np.array(to_rgb("#75808a"))
reflector_colour = np.array(to_rgb("#dfe5ec"))
# The drawn half-length of a polarization line, in mm.
line_half_width = 0.45
# The entering polarization's reach and the reflector's, in polarization half-lengths; the dashes along each entering
# line and the share of its place each fills; the reflector's thickness in mm and its opacity.
entering_reach, reflector_reach = 1.3, 1.4
n_dashes, dash_fill = 7, 0.6
reflector_thickness, reflector_opacity = 0.03, 0.3
# The widths of the polarization lines, the dashes and the beam, in pixels.
line_width, dash_width, beam_width = 2.2, 1.1, 0.85
# Each panel's view: elevation and azimuth in radians, azimuth zero looking along +y; its margin, a share of the
# larger side, and the pixels across it.
elevation, azimuth = np.radians(12.0), np.radians(35.0)
margin, n_pixels = 0.04, 300
# The beam's frame turned into the scene's, whose z is up: a half turn about the line along x + z through the origin
# carries the beam, along z, onto x, across the picture, and the polarizations' x onto z, up.
upright = ((sm.yz + sm.xy) * (np.pi / (2 * np.sqrt(2)))).exp()


def draw_isolator(fields: Field, depths: np.ndarray) -> plt.Figure:
    """`[media, setups, legs, depths]` electric fields: one panel per medium and setup, the beam running
    left to right, out to the reflector and back, each field drawn as its polarization line, with the
    entering polarization dashed at both ends; seen from aside and a little above, x up."""
    mv = fields.context.multivector
    time = -mv.t.dual()
    # Each field read by pairing it with the planes the directions x, y and z sweep through time.
    swept_planes = stack((mv.x.dual(), mv.y.dual(), mv.z.dual())) & time
    vectors = (fields[..., None] & swept_planes).to_array()               # [media, setups, legs, depths, 3]
    length = depths[-1]
    # The places along the beam, out and back, and its two ends, where the light enters and leaves.
    beam_places = np.stack([depths, length + depths])[..., None] * np.eye(3)[2]            # [legs, depths, 3]
    ends = np.array([0.0, 2 * length])[:, None] * np.eye(3)[2]                            # [2, 3]
    media, setups = vectors.shape[:2]

    # The reflector: a square across the beam at its far end, and the box it and the beam span, framed.
    reflector_half_width = reflector_reach * line_half_width
    square_corners = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]])                  # [4, 3]
    square = square_corners * reflector_half_width + length * np.eye(3)[2]                # [4, 3]
    corners = np.concatenate([square - length * np.eye(3)[2], square + length * np.eye(3)[2]])          # [8, 3]
    view = solids.view([upright >> sm.point(corners)], 0.0, elevation, azimuth, margin, n_pixels)
    reflector = solids.plates((upright >> sm.point(square))[None], reflector_thickness)
    beam = solids.rods(upright >> sm.point(ends[0]), upright >> sm.point(ends[1]), beam_width / 2 * view.pixel)[None]
    # Each dash's start and end along its line, as shares of the line's half-length from its middle.
    cells = np.linspace(-1, 1, n_dashes + 1)
    gap = (1 - dash_fill) * (cells[1] - cells[0]) / 2
    dash_starts, dash_ends = cells[:-1] + gap, cells[1:] - gap                             # [dashes]

    rows, columns = view.shape
    figure = plt.figure(figsize=(media * columns / 100, setups * rows / 100), dpi=100)
    for medium in range(media):
        for setup in range(setups):
            half_lines = line_half_width * vectors[medium, setup]                          # [legs, depths, 3]
            lines = solids.rods(upright >> sm.point(beam_places - half_lines),
                                upright >> sm.point(beam_places + half_lines),
                                line_width / 2 * view.pixel)                               # [legs, depths, 3]
            entering_half_line = entering_reach * line_half_width * vectors[medium, setup, 0, 0]
            dashes = solids.rods(upright >> sm.point(ends[:, None] + dash_starts[:, None] * entering_half_line),
                                 upright >> sm.point(ends[:, None] + dash_ends[:, None] * entering_half_line),
                                 dash_width / 2 * view.pixel)                              # [2, dashes, 3]
            parts = [(beam, beam_colour), (dashes.reshape(-1, 3), entering_colour),
                     *((lines[leg], colour) for leg, colour in enumerate(leg_colours))]
            image = solids.draw(parts, view, np.ones(3), inks=[],
                                translucent=[(reflector, reflector_colour, reflector_opacity)], meshes=[])
            ax = figure.add_axes((medium / media, 1 - (setup + 1) / setups, 1 / media, 1 / setups))
            show(ax, image, view)
    return figure
