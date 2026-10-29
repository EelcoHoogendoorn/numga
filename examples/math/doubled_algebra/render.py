"""A shear acting on a square, its grid and the two basis vectors."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon

from numga import stack

from examples.math.doubled_algebra import core

reference_colour = "#cbd2d8"
grid_colour = "#3885a0"
boundary_colour = "#cc793e"
fill_colour = "#fff1df"
basis_colours = ("#168b82", "#8050ad")
# The figure's size in inches, at a hundred pixels an inch, and the widths of its lines in points.
figure_size = (3.2, 2.2)
grid_width, reference_width, boundary_width = 0.6, 1.0, 1.5


# --- plumbing -------------------------------------------------------------------------
def coordinates(vectors: core.Vector) -> np.ndarray:
    """The x and y coefficients of vectors, read by pairing them with the antivectors complementing x and y."""
    mv = vectors.context.multivector
    return (vectors[..., None] & stack((mv.x, mv.y)).dual()).to_array()


def draw_shear(
    reference_grid: core.Vector,
    grid: core.Vector,
    reference_boundary: core.Vector,
    boundary: core.Vector,
    arrows: core.Vector,
) -> plt.Figure:
    """The reference square in grey and its sheared grid, with the images of the two basis vectors."""
    reference_lines = coordinates(reference_grid)
    lines = coordinates(grid)
    reference_corners = coordinates(reference_boundary)
    corners = coordinates(boundary)
    basis_images = coordinates(arrows)

    points = np.concatenate([
        reference_lines.reshape(-1, 2), lines.reshape(-1, 2),
        reference_corners, corners, basis_images, np.zeros((1, 2)),
    ])
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    margin = (upper - lower).max() * 0.08

    figure, ax = plt.subplots(figsize=figure_size, layout="constrained")
    ax.add_patch(Polygon(corners, closed=True, facecolor=fill_colour, edgecolor="none", zorder=0))
    ax.add_collection(LineCollection(reference_lines, colors=reference_colour, linewidths=grid_width, zorder=1))
    ax.add_patch(Polygon(reference_corners, closed=True, fill=False, edgecolor=reference_colour,
                         linewidth=reference_width, linestyle=(0, (3, 3)), zorder=1))
    ax.add_collection(LineCollection(lines, colors=grid_colour, linewidths=grid_width, zorder=2))
    ax.add_patch(Polygon(corners, closed=True, fill=False, edgecolor=boundary_colour, linewidth=boundary_width,
                         zorder=3))
    origins = np.zeros(len(basis_images))
    ax.quiver(origins, origins, *basis_images.T, color=basis_colours,
              angles="xy", scale_units="xy", scale=1, width=0.009,
              headwidth=4.2, headlength=5.2, zorder=4)
    ax.scatter(0, 0, s=10, color="#3b4146", zorder=5)
    ax.set(xlim=(lower[0] - margin, upper[0] + margin),
           ylim=(lower[1] - margin, upper[1] + margin), aspect="equal")
    ax.set_axis_off()
    return figure
