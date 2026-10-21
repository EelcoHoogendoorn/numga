"""A shear acting on a square, its grid and its two edge directions."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon

from examples.math.doubled_algebra import core

REFERENCE = "#cbd2d8"
GRID = "#3885a0"
BOUNDARY = "#cc793e"
FILL = "#fff1df"
DIRECTIONS = ("#168b82", "#8050ad")


# --- plumbing -------------------------------------------------------------------------
def draw_shear(
    reference_grid: core.Vector,
    grid: core.Vector,
    reference_boundary: core.Vector,
    boundary: core.Vector,
    arrows: core.Vector,
) -> plt.Figure:
    """The reference square in grey and its sheared grid with the two transformed edge directions."""
    layout = grid.algebra.subspace("x y")
    reference_lines = reference_grid.cast(layout).kernel
    lines = grid.cast(layout).kernel
    reference_corners = reference_boundary.cast(layout).kernel
    corners = boundary.cast(layout).kernel
    directions = arrows.cast(layout).kernel

    points = np.concatenate([
        reference_lines.reshape(-1, 2), lines.reshape(-1, 2),
        reference_corners, corners, directions, np.zeros((1, 2)),
    ])
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    margin = (upper - lower).max() * 0.08

    figure, ax = plt.subplots(figsize=(6.4, 4.4), layout="constrained")
    ax.add_patch(Polygon(corners, closed=True, facecolor=FILL, edgecolor="none", zorder=0))
    ax.add_collection(LineCollection(reference_lines, colors=REFERENCE, linewidths=0.9, zorder=1))
    ax.add_patch(Polygon(reference_corners, closed=True, fill=False, edgecolor=REFERENCE,
                         linewidth=1.5, linestyle=(0, (3, 3)), zorder=1))
    ax.add_collection(LineCollection(lines, colors=GRID, linewidths=0.9, zorder=2))
    ax.add_patch(Polygon(corners, closed=True, fill=False, edgecolor=BOUNDARY, linewidth=2.2, zorder=3))
    origins = np.zeros(len(directions))
    ax.quiver(origins, origins, *directions.T, color=DIRECTIONS,
              angles="xy", scale_units="xy", scale=1, width=0.009,
              headwidth=4.2, headlength=5.2, zorder=4)
    ax.scatter(0, 0, s=24, color="#3b4146", zorder=5)
    ax.set(xlim=(lower[0] - margin, upper[0] + margin),
           ylim=(lower[1] - margin, upper[1] + margin), aspect="equal")
    ax.set_axis_off()
    return figure
