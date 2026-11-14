"""Drawing for Penrose tilings renders selected lattice faces as thick and thin rhombi on the cut."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from examples.util.animation import capture
from examples.util.panels import offset_panel, overlay
from numga.render import Camera, contours, drawing, simplices
from numga.topology import Topology
from examples.math.penrose import core

# Colours of thick rhombi, thin rhombi, and tile borders.
thick_colour, thin_colour, edge_colour = (np.array(to_rgb(colour)) for colour in ("#f59e0b", "#38bdf8", "#1e293b"))

# Figure side lengths in inches at 100 dpi and border widths in pixels.
still_side, animation_side = 7.0, 3.5
still_edge_width, animation_edge_width = 0.8, 0.6


def cut_canvas(reading: core.Plane, view_extent: float, side: float) -> tuple[plt.Figure, plt.Axes, Camera]:
    """Borderless square figure, axes framed to `view_extent`, and affine camera reading points with `reading` (`[2] Plane+`)."""
    figure = plt.figure(figsize=(side, side), dpi=100)
    ax = figure.add_axes((0, 0, 1, 1))
    ax.set_xlim(-view_extent, view_extent)
    ax.set_ylim(-view_extent, view_extent)
    ax.set_aspect("equal")
    ax.set_axis_off()
    return figure, ax, offset_panel(ax, reading[0], reading[1])


def tile_picture(corners: core.Point, face_colours: np.ndarray, margins: core.Scalar, camera: Camera,
                 edge_width: float) -> np.ndarray:
    """Renders faces whose three `margins` (`[..., 10, 3] Scalar+`) are all positive as filled rhombi with stroked borders."""
    selected = np.all(margins > 0, axis=-1)                                   # [..., 10]
    n_tiles = int(np.count_nonzero(selected))
    # Split each selected four-corner rhombus into two triangles and draw their exterior border as the tile outline.
    tile_vertices = np.arange(4 * n_tiles).reshape(n_tiles, 4)                # [n_tiles, 4]
    halves = Topology(4 * n_tiles, tile_vertices[:, [[0, 1, 2], [0, 2, 3]]].reshape(-1, 3))
    border_edges = halves.border
    places = corners[selected].reshape(-1)                                    # [4 * n_tiles] Point-
    vertex_colours = np.repeat(face_colours[np.nonzero(selected)[1]], 4, axis=0)  # [4 * n_tiles, 3]
    return drawing.composited([
        contours.picture(places, halves, np.zeros(4 * n_tiles), vertex_colours, camera),
        simplices.strokes(places, border_edges, edge_colour, edge_width, camera),
    ])


def tiling_figure(reading: core.Plane, corners: core.Point, areas: core.Scalar, view_extent: float,
                  margins: core.Scalar) -> plt.Figure:
    """Static figure of the rhombus tiling selected by `margins` (`[..., 10, 3] Scalar+`), coloured by thick versus thin `areas`."""
    face_colours = np.where((areas > areas.mean())[:, None], thick_colour, thin_colour)  # [10, 3]
    figure, ax, camera = cut_canvas(reading, view_extent, still_side)
    overlay(ax, [tile_picture(corners, face_colours, margins, camera, still_edge_width)])
    return figure


def phason_frames(reading: core.Plane, corners: core.Point, areas: core.Scalar, view_extent: float,
                  margin_states: Iterable[core.Scalar]) -> Iterator[np.ndarray]:
    """Yields RGB frames as `margin_states` steps through perpendicular shifts of the cut on a reusable canvas."""
    face_colours = np.where((areas > areas.mean())[:, None], thick_colour, thin_colour)  # [10, 3]
    figure, ax, camera = cut_canvas(reading, view_extent, animation_side)
    shown = overlay(ax, [np.zeros(camera.shape + (4,))])
    for margins in margin_states:
        shown.set_data(tile_picture(corners, face_colours, margins, camera, animation_edge_width))
        yield capture(figure)
    plt.close(figure)
