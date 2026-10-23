"""Drawing for Penrose tilings: the tiles the cut selects, laid flat onto it, thick and thin."""

from __future__ import annotations

from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection

from examples.util.animation import capture
from examples.math.penrose import core

# The colours of thick and thin tiles, and of their edges.
THICK = "#f59e0b"
THIN = "#38bdf8"
EDGE = "#1e293b"


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


def tiles(ax: plt.Axes, polygons: np.ndarray, colours: np.ndarray, margins: core.Scalar):
    """The faces whose three margins are all positive, `polygons [points, faces, 4, 2]`."""
    chosen = np.all(margins.to_array() > 0, axis=-1)                         # [points, faces]
    ax.add_collection(PolyCollection(polygons[chosen], facecolors=colours[np.nonzero(chosen)[1]],
                                     edgecolors=EDGE, linewidths=0.6))


def draw_tiling(reading: core.Plane, corners: core.Antivector, areas: core.Scalar, margins: core.Scalar,
                extent: float) -> plt.Figure:
    """The tiles the cut selects among the faces with these corners `[points, faces, 4]`."""
    shapes = areas.to_array()
    colours = np.where(shapes > shapes.mean(), THICK, THIN)                  # [faces]
    figure, ax = canvas(extent)
    tiles(ax, flat(reading, corners), colours, margins)
    return figure


def animate_flips(reading: core.Plane, corners: core.Antivector, areas: core.Scalar, frames: Iterable[core.Scalar],
                  extent: float) -> list[np.ndarray]:
    """The tiles each frame's margins select."""
    shapes = areas.to_array()
    colours = np.where(shapes > shapes.mean(), THICK, THIN)                  # [faces]
    polygons = flat(reading, corners)                                        # [points, faces, 4, 2]
    images = []
    for margins in frames:
        figure, ax = canvas(extent)
        tiles(ax, polygons, colours, margins)
        images.append(capture(figure))
        plt.close(figure)
    return images
