"""Polarizations along a beam, out to a reflector and back, unfolded into one straight path."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from numga import concatenate, stack
from examples.orientation.optical_rotation.core import Direction, Field, mv
from examples.util.wgsl import show, solids

# The observer sees the point where its time line meets the spatial slice.
space_slice = mv.t ^ (Direction & -mv.t.dual())

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


# --- plumbing -------------------------------------------------------------------------
def draw_isolator(fields: Field, depths: np.ndarray) -> plt.Figure:
    """`[media, setups, legs, depths]` electric fields: one panel per medium and setup, the beam running
    left to right, out to the reflector and back, each field drawn as its polarization line, with the
    entering polarization dashed at both ends; seen from aside and a little above, x up."""
    # Each field's polarization: the plane it sweeps through time, met with the observer's slice of space.
    polarizations = fields.dual_inverse() ^ mv.t                              # [media, setups, legs, depths] Direction-
    # The direction of the beam, its length, the places along it, out and back, and its two ends, where the light
    # enters and leaves.
    beam = mv.z.dual()                                                        # [] Direction-
    length = depths[-1]
    beam_places = np.stack([depths, length + depths]) * beam                  # [legs, depths] Direction-
    ends = np.array([0.0, 2 * length]) * beam                                 # [2] Direction-
    media, setups = fields.shape[:2]

    # The reflector: a square across the beam at its far end, and the box it and the beam span, framed.
    reflector_half_width = reflector_reach * line_half_width
    square = (reflector_half_width * (mv.x.dual() * np.array([-1, 1, 1, -1]) + mv.y.dual() * np.array([-1, -1, 1, 1]))
              + length * beam)                                                # [corners] Direction-
    corners = concatenate([square - length * beam, square + length * beam])   # [2 * corners] Direction-
    scene_corners = solids.origin + (upright >> solids.context.include_from_algebra(space_slice(corners)))
    scene_square = solids.origin + (upright >> solids.context.include_from_algebra(space_slice(square)))
    beam_ends = solids.origin + (upright >> solids.context.include_from_algebra(space_slice(ends)))
    stations = solids.origin + (upright >> solids.context.include_from_algebra(space_slice(beam_places)))
    directions = upright >> solids.context.include_from_algebra(space_slice(polarizations))
    view = solids.view([scene_corners], 0.0, elevation, azimuth, margin, n_pixels)
    reflector = solids.plates(scene_square[None], reflector_thickness)
    beam_rod = solids.rods(beam_ends[0], beam_ends[1], beam_width / 2 * view.pixel)[None]
    # Each dash's start and end along its line, as shares of the line's half-length from its middle.
    cells = np.linspace(-1, 1, n_dashes + 1)
    gap = (1 - dash_fill) * (cells[1] - cells[0]) / 2
    dash_starts, dash_ends = cells[:-1] + gap, cells[1:] - gap                             # [dashes]

    rows, columns = view.shape
    figure = plt.figure(figsize=(media * columns / 100, setups * rows / 100), dpi=100)
    for medium in range(media):
        for setup in range(setups):
            half_lines = line_half_width * directions[medium, setup]         # [legs, depths] Point-
            lines = solids.rods(stations - half_lines, stations + half_lines,
                                line_width / 2 * view.pixel)                               # [legs, depths, 3]
            entering_half_line = entering_reach * line_half_width * directions[medium, setup, 0, 0]   # [] Point-
            dashes = solids.rods(beam_ends[:, None] + dash_starts * entering_half_line,
                                 beam_ends[:, None] + dash_ends * entering_half_line,
                                 dash_width / 2 * view.pixel)                              # [2, dashes, 3]
            parts = [(beam_rod, beam_colour), (dashes.reshape(-1, 3), entering_colour),
                     *((lines[leg], colour) for leg, colour in enumerate(leg_colours))]
            image = solids.draw(parts, view, np.ones(3), inks=[],
                                translucent=[(reflector, reflector_colour, reflector_opacity)], meshes=[])
            ax = figure.add_axes((medium / media, 1 - (setup + 1) / setups, 1 / media, 1 / setups))
            show(ax, image, view)
    return figure
