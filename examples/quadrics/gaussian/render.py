"""Implicit contour rendering draws the fitted Gaussian density, point cloud, and one-sigma quadric."""

from __future__ import annotations

from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb, to_rgba

from examples.quadrics.gaussian.core import Point, Polarity, Scalar, mv
from examples.util import charts
from examples.util.panels import overlay, panel, room
from numga.render import Camera, contours, simplices
from numga.topology import Topology

view_bounds = np.array([[-5.0, -5.0], [6.0, 4.0]])
figure_width_px = 300
sample_side_px = 3
sample_ink = np.array(to_rgb("#26364a"))
sample_opacity = 0.4
ellipsoid_ink = np.array(to_rgba("#e56b24"))
ellipsoid_width_px = 1.5
cross_ink = np.full(3, 0.8)


@charts.style()
def gaussian_figure(points: Point, density: Callable[[Point], Scalar], ellipsoid: Polarity) -> plt.Figure:
    """Pixel-corner evaluations shade the Gaussian density behind the point cloud (`[...] Point-`) and the zero contour of the one-sigma quadric."""
    (left, bottom), (right, top) = view_bounds
    fig, ax = charts.subplots(figure_width_px, round(figure_width_px * (top - bottom) / (right - left)))
    ax.set_position((0, 0, 1, 1))
    ax.set(xlim=(left, right), ylim=(bottom, top))
    n_columns, n_rows = room(ax)
    # Evaluate the density and quadric form at every pixel corner from top to bottom.
    x_grid = np.linspace(left, right, n_columns + 1)
    y_grid = np.linspace(top, bottom, n_rows + 1)
    grid_points = mv.point(np.stack(np.meshgrid(x_grid, y_grid), axis=-1))     # [n_rows + 1, n_columns + 1] Point-
    shades = plt.get_cmap("Blues")(np.clip(density(grid_points).to_array(), 0, 1))[..., :3]
    quadric_values = ellipsoid(grid_points) & grid_points                      # [n_rows + 1, n_columns + 1] Scalar+
    flat_points = grid_points.reshape(-1)                                      # [(n_rows + 1) * (n_columns + 1)] Point-
    flat_values = quadric_values.reshape(-1).to_array()
    flat_shades = shades.reshape(-1, 3)
    pixel_grid = Topology.grid(n_rows + 1, n_columns + 1)
    camera = panel(ax, mv.x, mv.y, mv.w)
    density_layer = contours.picture(flat_points, pixel_grid, flat_values, flat_shades, camera)
    cloud_layer = simplices.markers(points, sample_ink, sample_side_px / 2, camera, opacity=sample_opacity)
    outline = contours.Level(0.0, ellipsoid_ink, ellipsoid_width_px)
    ellipse_layer = contours.picture(
        flat_points, pixel_grid, flat_values, flat_shades, camera, levels=(outline,), opacity=0.0,
    )
    overlay(ax, [density_layer, cloud_layer, origin_cross(ax, camera, "x", "y"), ellipse_layer])
    return fig


def origin_cross(ax: plt.Axes, camera: Camera, horizontal_label: str, vertical_label: str) -> np.ndarray:
    """Faint coordinate lines through the origin mark the horizontal and vertical axes across the panel."""
    (left, right), (bottom, top) = ax.get_xlim(), ax.get_ylim()
    axis_ends = mv.point([[left, 0], [right, 0], [0, bottom], [0, top]])       # [4] Point-
    ax.annotate(
        horizontal_label, (1, 0), xycoords=("axes fraction", "data"), xytext=(-4, -3),
        textcoords="offset points", ha="right", va="top", color="0.4",
    )
    ax.annotate(
        vertical_label, (0, 1), xycoords=("data", "axes fraction"), xytext=(4, -4),
        textcoords="offset points", ha="left", va="top", color="0.4",
    )
    ax.set_axis_off()
    return simplices.strokes(axis_ends, Topology.chains([2, 2]), cross_ink, charts.FRAME_PX, camera, cap=simplices.Cap.BUTT)
