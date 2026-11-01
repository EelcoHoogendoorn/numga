"""Drawing for Penrose tilings: the tiles the cut selects, laid flat onto it, thick and thin."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from examples.util.animation import capture
from examples.util.wgsl import contours, simplices
from examples.util.wgsl.camera import Camera, panel
from examples.math.penrose import core

# The colours of thick and thin tiles, and of their edges.
thick_colour, thin_colour, edge_colour = (np.array(to_rgb(colour)) for colour in ("#f59e0b", "#38bdf8", "#1e293b"))

# The sides of the still and of the animation, in inches at 100 dpi, and the widths of their edges in pixels.
still_side, animation_side = 7.0, 3.5
still_edge_width, animation_edge_width = 0.8, 0.6


# --- plumbing -------------------------------------------------------------------------
def canvas(reading: core.Plane, extent: float, side: float) -> tuple[plt.Figure, plt.Axes, Camera]:
    """The figure, its axes on the cut, and their camera reading points by the two reading planes."""
    figure = plt.figure(figsize=(side, side), dpi=100)
    ax = figure.add_axes((0, 0, 1, 1))
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_aspect("equal")
    ax.set_axis_off()
    return figure, ax, panel(ax, reading[0], reading[1])


def tiles(corners: core.Antivector, colours: np.ndarray, margins: core.Scalar, camera: Camera,
          width: float) -> np.ndarray:
    """The faces whose three margins are all positive, among those with these corners `[points, faces, 4]`, each in
    its colour `[faces, 3]`, with their edges."""
    chosen = np.all(margins > 0, axis=-1)                                    # [points, faces]
    n_tiles = np.count_nonzero(chosen)
    # Each tile's four corners its own, two triangles of them, and the loop of its sides, their border.
    own = np.arange(4 * n_tiles).reshape(n_tiles, 4)
    halves = simplices.Topology(4 * n_tiles, own[:, [[0, 1, 2], [0, 2, 3]]].reshape(-1, 3))
    sides = halves.border
    places = corners[chosen].reshape(-1)                                     # [tiles * 4] Antivector-
    return simplices.composited([
        contours.picture(places, halves, np.zeros(4 * n_tiles), np.repeat(colours[np.nonzero(chosen)[1]], 4, axis=0),
                         camera),
        simplices.strokes(places, sides, edge_colour, width, camera)])


def draw_tiling(reading: core.Plane, corners: core.Antivector, areas: core.Scalar, margins: core.Scalar,
                extent: float) -> plt.Figure:
    """The tiles the cut selects among the faces with these corners `[points, faces, 4]`."""
    colours = np.where((areas > areas.mean())[:, None], thick_colour, thin_colour)   # [faces, 3]
    figure, ax, camera = canvas(reading, extent, still_side)
    simplices.overlay(ax, [tiles(corners, colours, margins, camera, still_edge_width)])
    return figure


def animate_flips(reading: core.Plane, corners: core.Antivector, areas: core.Scalar, frames: Iterable[core.Scalar],
                  extent: float) -> Iterator[np.ndarray]:
    """The tiles each frame's margins select, on a canvas laid out once."""
    colours = np.where((areas > areas.mean())[:, None], thick_colour, thin_colour)   # [faces, 3]
    figure, ax, camera = canvas(reading, extent, animation_side)
    shown = simplices.overlay(ax, [np.zeros(camera.shape + (4,))])
    for margins in frames:
        shown.set_data(tiles(corners, colours, margins, camera, animation_edge_width))
        yield capture(figure)
    plt.close(figure)
