"""Drawing and coordinate readout for the Gaussian fit."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb, to_rgba

from collections.abc import Callable

from examples.quadrics.gaussian.core import Point, Polarity, Scalar, mv
from examples.util import charts
from examples.util.wgsl import contours, room, simplices
from examples.util.wgsl.camera import Camera, flat_panel

# The box the fit is shown over, along x and y, and the figure's width in pixels, for one cloud and one ellipse.
BOX = np.array([[-5.0, -5.0], [6.0, 4.0]])
WIDTH = 300
# The side of a sample's dot in pixels, small for the cloud's density to show; its ink and how much it covers.
SAMPLE_SIDE = 3
SAMPLE_INK = np.array(to_rgb("#26364a"))
SAMPLE_OPACITY = 0.4
# The ellipsoid's ink and its width in pixels.
ELLIPSOID_INK = np.array(to_rgba("#e56b24"))
ELLIPSOID_WIDTH = 1.5
# The cross's ink, faint.
CROSS_INK = np.full(3, 0.8)


# --- plumbing -------------------------------------------------------------------------
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
    places = corners.reshape(-1)                                                # [(rows + 1) * (columns + 1)] Point-
    grid = contours.Mesh.grid(*form.shape)
    camera = flat_panel(ax, places.algebra)
    shaded = contours.picture(places, grid, form.reshape(-1), shades.reshape(-1, 3), camera)
    cloud = simplices.markers(points, SAMPLE_INK, SAMPLE_SIDE / 2, camera, opacity=SAMPLE_OPACITY)
    # The ellipsoid alone, over the cloud and the cross: its zero level on a clear picture.
    outline = contours.Level(0.0, ELLIPSOID_INK, ELLIPSOID_WIDTH)
    ellipse = contours.picture(places, grid, form.reshape(-1), shades.reshape(-1, 3), camera, levels=(outline,),
                               opacity=0.0)
    simplices.overlay(ax, [shaded, cloud, cross(ax, camera, "x", "y"), ellipse])
    return fig


def cross(ax, camera: Camera, across: str, up: str) -> np.ndarray:
    """No scales: a faint cross through the origin, the coordinate planes through it from edge to edge of the panel,
    each named at its end; the cross's picture."""
    (left, right), (bottom, top) = ax.get_xlim(), ax.get_ylim()
    ends = mv.point([[left, 0], [right, 0], [0, bottom], [0, top]])            # [4] Point-
    ax.annotate(across, (1, 0), xycoords=("axes fraction", "data"), xytext=(-4, -3), textcoords="offset points",
                ha="right", va="top", color="0.4")
    ax.annotate(up, (0, 1), xycoords=("data", "axes fraction"), xytext=(4, -4), textcoords="offset points",
                ha="left", va="top", color="0.4")
    ax.set_axis_off()
    return simplices.strokes(ends, simplices.Mesh.chains([2, 2]), CROSS_INK, charts.FRAME_PX, camera,
                             cap=simplices.Cap.BUTT)
