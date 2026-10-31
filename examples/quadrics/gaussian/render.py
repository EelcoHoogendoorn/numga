"""Drawing and coordinate readout for the Gaussian fit."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba

from numga import stack
from collections.abc import Callable

from examples.quadrics.gaussian.core import Point, Polarity, Scalar, mv
from examples.util import charts
from examples.util.wgsl import contours, room, solids

# The box the fit is shown over, along x and y, and the figure's width in pixels, for one cloud and one ellipse.
BOX = np.array([[-5.0, -5.0], [6.0, 4.0]])
WIDTH = 300
# The side of a sample's dot in pixels, small for the cloud's density to show.
SAMPLE_SIDE = 3
# The ellipsoid's ink and its width in pixels.
ELLIPSOID_INK = np.array(to_rgba("#e56b24"))
ELLIPSOID_WIDTH = 1.5


# --- plumbing -------------------------------------------------------------------------
def euclidean(points: Point) -> np.ndarray:
    """Coordinates of points, `[..., 2]`: their pairings with the planes x = 0 and y = 0, over their weights."""
    return ((stack((mv.x, mv.y)) & points[..., None]) / (mv.w & points)[..., None]).to_array()


@charts.style()
def draw_gaussian(points: Point, density: Callable[[Point], Scalar], ellipsoid: Polarity) -> plt.Figure:
    """The density over the pixels, the point cloud, and the 1σ ellipsoid, orange, the zero contour of its form."""
    (left, bottom), (right, top) = BOX
    fig, ax = charts.subplots(WIDTH, round(WIDTH * (top - bottom) / (right - left)))
    ax.set_position((0, 0, 1, 1))
    ax.set(xlim=(left, right), ylim=(bottom, top))
    columns, rows = room(ax)
    # One point at each corner of the pixels, rows down from the top.
    across = np.linspace(left, right, columns + 1)
    down = np.linspace(top, bottom, rows + 1)
    corners = mv.point(np.stack(np.meshgrid(across, down), axis=-1))            # [rows + 1, columns + 1] Point-
    shades = plt.get_cmap("Blues")(np.clip(density(corners).to_array(), 0, 1))[..., :3]
    form = (ellipsoid(corners) & corners).to_array()                            # [rows + 1, columns + 1]
    places = solids.context.include_from_algebra(corners.reshape(-1))         # [(rows + 1) * (columns + 1)] Point-
    grid = contours.Mesh.grid(*form.shape)
    contours.overlay(ax, places, grid, form.reshape(-1), shades.reshape(-1, 3))
    sample_xy = euclidean(points)
    ax.scatter(sample_xy[:, 0], sample_xy[:, 1], s=charts.points(SAMPLE_SIDE) ** 2, c="#26364a", alpha=0.4,
               linewidths=0)
    cross(ax, "x", "y")
    # The ellipsoid alone, over the cloud and the cross: its zero level on a clear picture.
    outline = contours.Level(0.0, ELLIPSOID_INK, ELLIPSOID_WIDTH)
    contours.overlay(ax, places, grid, form.reshape(-1), shades.reshape(-1, 3), levels=(outline,),
                     opacity=0.0).set_zorder(2)
    return fig


def cross(ax, across: str, up: str) -> None:
    """No scales: a faint cross through the origin over the density, the coordinate planes through it, each named at its
    end."""
    ax.axhline(0, color="0.8", linewidth=charts.FRAME_WIDTH, zorder=1.5)
    ax.axvline(0, color="0.8", linewidth=charts.FRAME_WIDTH, zorder=1.5)
    ax.annotate(across, (1, 0), xycoords=("axes fraction", "data"), xytext=(-4, -3), textcoords="offset points",
                ha="right", va="top", color="0.4")
    ax.annotate(up, (0, 1), xycoords=("data", "axes fraction"), xytext=(4, -4), textcoords="offset points",
                ha="left", va="top", color="0.4")
    ax.set_axis_off()
