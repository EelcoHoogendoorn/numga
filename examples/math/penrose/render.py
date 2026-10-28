"""Drawing for Penrose tilings: the tiles the cut selects, laid flat onto it, thick and thin."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection

from examples.util.animation import capture
from examples.math.penrose import core

# The colours of thick and thin tiles, and of their edges.
thick_colour = "#f59e0b"
thin_colour = "#38bdf8"
edge_colour = "#1e293b"


# --- plumbing -------------------------------------------------------------------------
def flat(reading: core.Plane, points: core.Antivector) -> np.ndarray:
    """The coordinates of points in the cut, read by pairing them with the two reading planes, on the
    last axis."""
    return (reading & points[..., None]).to_array()


def canvas(extent: float) -> tuple[plt.Figure, plt.Axes]:
    figure = plt.figure(figsize=(7.0, 7.0), dpi=100)
    ax = figure.add_axes((0, 0, 1, 1))
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_aspect("equal")
    ax.set_axis_off()
    return figure, ax


def selected(polygons: np.ndarray, colours: np.ndarray, margins: core.Scalar) -> tuple[np.ndarray, np.ndarray]:
    """The faces whose three margins are all positive, `polygons [points, faces, 4, 2]`, and their colours."""
    chosen = np.all(margins > 0, axis=-1)                                    # [points, faces]
    return polygons[chosen], colours[np.nonzero(chosen)[1]]


def tiles(ax: plt.Axes, polygons: np.ndarray, colours: np.ndarray, margins: core.Scalar):
    """The faces whose three margins are all positive, `polygons [points, faces, 4, 2]`."""
    faces, shades = selected(polygons, colours, margins)
    ax.add_collection(PolyCollection(faces, facecolors=shades, edgecolors=edge_colour, linewidths=0.6))


def draw_tiling(reading: core.Plane, corners: core.Antivector, areas: core.Scalar, margins: core.Scalar,
                extent: float) -> plt.Figure:
    """The tiles the cut selects among the faces with these corners `[points, faces, 4]`."""
    colours = np.where(areas > areas.mean(), thick_colour, thin_colour)      # [faces]
    figure, ax = canvas(extent)
    tiles(ax, flat(reading, corners), colours, margins)
    return figure


def animate_flips(reading: core.Plane, corners: core.Antivector, areas: core.Scalar, frames: Iterable[core.Scalar],
                  extent: float) -> Iterator[np.ndarray]:
    """The tiles each frame's margins select, on a canvas laid out once."""
    colours = np.where(areas > areas.mean(), thick_colour, thin_colour)      # [faces]
    polygons = flat(reading, corners)                                        # [points, faces, 4, 2]
    figure, ax = canvas(extent)
    shown = ax.add_collection(PolyCollection([], edgecolors=edge_colour, linewidths=0.6))
    for margins in frames:
        faces, shades = selected(polygons, colours, margins)
        shown.set_verts(faces)
        shown.set_facecolor(shades)
        yield capture(figure)
    plt.close(figure)
