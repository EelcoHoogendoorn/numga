"""Draws a planar square, its reference grid, their sheared images and the sheared basis vectors."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from numga import concatenate, stack
from numga.render import contours, simplices
from numga.topology import Topology
from examples.math.doubled_algebra import core
from examples.util.panels import offset_panel, overlay

mv = core.mv
reference_colour, grid_colour, boundary_colour, fill_colour, dot_colour = (
    np.array(to_rgb(colour)) for colour in ("#cbd2d8", "#3885a0", "#cc793e", "#fff1df", "#3b4146")
)
basis_colours = np.array([to_rgb("#168b82"), to_rgb("#8050ad")])
# Figure size in inches at 100 dpi, stroke widths, reference dash pattern, chevron length, and origin dot radius in px.
figure_size = (3.2, 2.2)
grid_width, reference_width, boundary_width, arrow_width = 1.0, 1.4, 2.0, 2.5
reference_dashes = (4.0, 4.0)
arrow_chevron, dot_radius = 10.0, 2.2

# Pairing a vector on the left with the duals of x and y reads out its x and y coefficients.
readers = stack((mv.x, mv.y)).dual()                                           # [2] Antivector-


def shear_figure(
    reference_grid: core.Vector,
    grid: core.Vector,
    reference_boundary: core.Vector,
    boundary: core.Vector,
    arrows: core.Vector,
) -> plt.Figure:
    """Renders the grey reference square, its sheared grid and boundary, and the images of the two basis vectors."""
    origin = 0 * arrows[0]                                                     # [] Vector+
    scene_vectors = concatenate([
        reference_grid.reshape(-1), grid.reshape(-1), reference_boundary, boundary, arrows, origin[None],
    ])                                                                         # [...] Vector+
    # Bound the extent across every drawn element with a uniform margin around it.
    readings = scene_vectors & readers[:, None]                                # [2, ...] Scalar+
    lower, upper = readings.min(axis=-1), readings.max(axis=-1)                # [2] Scalar+, [2] Scalar+
    margin = (upper - lower).max() * 0.08                                      # [] Scalar+
    limits = stack((lower - margin, upper + margin), axis=-1).to_array()       # [2, 2]

    figure, ax = plt.subplots(figsize=figure_size, layout="constrained")
    ax.set(xlim=limits[0], ylim=limits[1], aspect="equal")
    ax.set_axis_off()
    # The camera reads vectors directly by pairing them against the duals of x and y.
    camera = offset_panel(ax, *readers)
    n_grid_lines = int(np.prod(grid.shape[:-1]))
    n_boundary = len(boundary)
    n_arrows = len(arrows)
    lines = Topology.chains([grid.shape[-1]] * n_grid_lines)
    square = Topology(n_boundary, np.array([[0, 1, 2], [0, 2, 3]]))
    # Each sheared basis vector is drawn from the origin as a two-point stroke ending in a chevron.
    arrow_ends = stack((origin.broadcast_to(arrows.shape), arrows), axis=-1).reshape(-1)  # [2 * n_arrows] Vector+
    overlay(ax, [
        contours.picture(boundary, square, np.zeros(n_boundary), fill_colour, camera),
        simplices.strokes(reference_grid.reshape(-1), lines, reference_colour, grid_width, camera),
        simplices.strokes(reference_boundary, Topology.loops([n_boundary]), reference_colour,
                          reference_width, camera, dashes=reference_dashes),
        simplices.strokes(grid.reshape(-1), lines, grid_colour, grid_width, camera),
        simplices.strokes(boundary, Topology.loops([n_boundary]), boundary_colour, boundary_width, camera),
        simplices.strokes(arrow_ends, Topology.chains([2] * n_arrows), basis_colours, arrow_width,
                          camera, chevrons=np.ones(n_arrows), chevron=arrow_chevron),
        simplices.markers(origin[None], dot_colour, dot_radius, camera),
    ])
    return figure
