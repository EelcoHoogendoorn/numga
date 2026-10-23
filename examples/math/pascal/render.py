"""Drawing for Pascal's theorem, on the GPU by `planar`: the conic, the curve where it vanishes; the sides of the
hexagon and the line through the crossings of opposite sides, whole across the view; the hexagon, and its corners and
the crossings as discs."""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from examples.util.wgsl import image_figure
from examples.util.wgsl import planar
from examples.util.wgsl.planar import Curves, Lines, Marker, Marks, Polygons, window
from examples.math.pascal import core

# The view: x from -2.5 to 4.5, y from -2.5 to 5.5, and the rows of pixels down it.
BOX = np.array([[-2.5, -2.5], [4.5, 5.5]])
ROWS = 590
# Colours of the three pairs of opposite sides and their crossings.
PAIRS = np.array(["#1f77b4", "#2a9d4a", "#9b59b6"])
# Pixels a point, at 100 dots per inch. The widths of the conic, of the sides and of the line through the crossings;
# the areas of the marks of the corners, of the sixth corner and of the crossings, in square points, and the width of
# their outlines.
POINT = 100 / 72
CONIC_WIDTH, SIDE_WIDTH, PASCAL_WIDTH = 1.6, 0.8, 2.0
CORNER_AREA, SIXTH_AREA, CROSSING_AREA, OUTLINE = 28.0, 60.0, 55.0, 1.5


def rgb(colours) -> np.ndarray:
    return np.array([mcolors.to_rgb(colour) for colour in np.atleast_1d(colours)])


def disc(points: core.Point, area: float, colour) -> Marks:
    """Points as discs of the area, with an outline in their colour."""
    return Marks(points, Marker.DISC, 0.0, (np.sqrt(area) / 2 + OUTLINE / 2) * POINT, rgb(colour), 1.0)


def picture(shape: core.Conic, hexagon: core.Point, crossing: core.Point) -> np.ndarray:
    """The hexagon under the conic, its sides in the colours of their pairs and the line through the crossings; its
    corners, the sixth orange, and the crossings, ringed in the line's colour: RGB bytes."""
    sides = hexagon & hexagon[core.NEXT]                                       # [6] Plane
    pascal_line = crossing[0] & crossing[2]                                    # [] Plane
    ring = np.sqrt(CROSSING_AREA) / 2
    layers = [Polygons(hexagon[None], rgb("0.85"), 0.5),
              Curves(shape[None], CONIC_WIDTH * POINT, rgb("0.25"), 1.0),
              Lines(sides, SIDE_WIDTH * POINT, rgb(np.tile(PAIRS, 2)), 0.6),
              Lines(pascal_line[None], PASCAL_WIDTH * POINT, rgb("#c0392b"), 1.0),
              disc(hexagon, CORNER_AREA, "0.15"),
              disc(hexagon[-1:], SIXTH_AREA, "#e67e22"),
              Marks(crossing, Marker.DISC, 0.0, ring * POINT, rgb(PAIRS), 1.0),
              Marks(crossing, Marker.DISC, (ring - OUTLINE / 2) * POINT, (ring + OUTLINE / 2) * POINT, rgb("#c0392b"),
                    1.0)]
    return planar.draw(window(BOX[:, 0], BOX[:, 1], ROWS), layers, np.ones(3))


def draw(shape: core.Conic, hexagon: core.Point, crossing: core.Point) -> plt.Figure:
    """Six points on a conic as a hexagon, its opposite sides in matching colours, and the three
    points where they meet on one line."""
    return image_figure(picture(shape, hexagon, crossing))


def animate(scenes: list[tuple[core.Conic, core.Point, core.Point]]) -> Iterator[np.ndarray]:
    """One frame per scene."""
    for scene in scenes:
        yield picture(*scene)
