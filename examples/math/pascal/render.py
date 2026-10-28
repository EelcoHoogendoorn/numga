"""Drawing for Pascal's theorem, on the GPU by `planar`: the quadric, the curve where it vanishes; the sides of the
hexagon and the line through the crossings of opposite sides, whole across the view; the hexagon, and its corners and
the crossings as discs."""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from examples.util.wgsl import image_figure
from examples.util.wgsl import planar
from examples.util.wgsl.planar import Curves, Flats, Marker, Marks, Polygons, window
from examples.math.pascal import core

# The view: x from -2.5 to 4.5, y from -2.5 to 5.5, and the rows of pixels down it.
box = np.array([[-2.5, -2.5], [4.5, 5.5]])
rows = 320
# Colours of the three pairs of opposite sides and their crossings.
pair_colours = np.array(["#1f77b4", "#2a9d4a", "#9b59b6"])
# Pixels a point. The widths of the quadric, of the sides and of the line through the crossings; the areas of the marks
# of the corners, of the sixth corner and of the crossings, in square points, and the width of their outlines.
point = 0.75
quadric_width, side_width, pascal_width = 1.6, 1.1, 2.0
corner_area, sixth_area, crossing_area, outline = 28.0, 60.0, 55.0, 1.5


def rgb(colours) -> np.ndarray:
    return np.array([mcolors.to_rgb(colour) for colour in np.atleast_1d(colours)])


def disc(points: core.Point, area: float, colour) -> Marks:
    """Points as discs of the area, with an outline in their colour."""
    return Marks(points, Marker.DISC, 0.0, (np.sqrt(area) / 2 + outline / 2) * point, rgb(colour), 1.0)


def picture(shape: core.Quadric, hexagon: core.Point, sides: core.Line, crossing: core.Crossing,
            pascal_line: core.Line) -> np.ndarray:
    """The hexagon under the quadric, its sides in the colours of their pairs and the line through the crossings; its
    corners, the sixth orange, and the crossings, ringed in the line's colour: RGB bytes."""
    ring = np.sqrt(crossing_area) / 2
    layers = [Polygons(hexagon[None], rgb("0.85"), 0.5),
              Curves(shape[None], quadric_width * point, rgb("0.25"), 1.0),
              Flats(sides, side_width * point, rgb(np.tile(pair_colours, 2)), 0.6),
              Flats(pascal_line[None], pascal_width * point, rgb("#c0392b"), 1.0),
              disc(hexagon, corner_area, "0.15"),
              disc(hexagon[-1:], sixth_area, "#e67e22"),
              Marks(crossing, Marker.DISC, 0.0, ring * point, rgb(pair_colours), 1.0),
              Marks(crossing, Marker.DISC, (ring - outline / 2) * point, (ring + outline / 2) * point, rgb("#c0392b"),
                    1.0)]
    return planar.draw(window(box[:, 0], box[:, 1], rows), layers, np.ones(3))


def draw(shape: core.Quadric, hexagon: core.Point, sides: core.Line, crossing: core.Crossing,
         pascal_line: core.Line) -> plt.Figure:
    """Six points on a quadric as a hexagon, its opposite sides in matching colours, and the three
    points where they meet on one line."""
    return image_figure(picture(shape, hexagon, sides, crossing, pascal_line))


def animate(scenes: list[tuple[core.Quadric, core.Point, core.Line, core.Crossing, core.Line]]) -> Iterator[np.ndarray]:
    """One frame per scene."""
    for scene in scenes:
        yield picture(*scene)
