"""Drawing for Poncelet's porism, on the GPU by `planar`: the two pairs of conics side by side, the ellipse the curve
where it vanishes and each inner conic the curve its tangents touch, and the path of five sides from the same start
about each inner conic."""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from examples.util.wgsl import image_figure
from examples.util.wgsl import planar
from examples.util.wgsl.planar import SOLID, Cap, Curves, Envelopes, Marker, Marks, Strokes, window
from examples.math.poncelet import core

# The view: x from -1.8 to 2.6, y from -1.9 to 1.5; the rows of pixels down it, and the columns between the two.
BOX = np.array([[-1.8, -1.9], [2.6, 1.5]])
ROWS = 369
GAP = 15
# Pixels a point, at 100 dots per inch. The widths of the ellipse, of the inner conic and of the path; the areas of
# the marks of the start and of the ring about the last vertex, in square points, and the width of their outlines.
POINT = 100 / 72
OUTER_WIDTH, INNER_WIDTH, PATH_WIDTH = 1.6, 1.4, 1.4
START_AREA, LAST_AREA, OUTLINE = 50.0, 110.0, 1.5


def rgb(colour: str) -> np.ndarray:
    return np.array(mcolors.to_rgb(colour))


def panel(outer: core.Conic, inner: core.Envelope, vertices: core.Point) -> np.ndarray:
    """The ellipse, an inner conic and the path of sides about it, the start orange and the last vertex ringed: RGB
    bytes."""
    start, last = np.sqrt(START_AREA) / 2, np.sqrt(LAST_AREA) / 2
    layers = [Curves(outer[None], OUTER_WIDTH * POINT, rgb("0.25"), 1.0),
              Envelopes(inner[None], INNER_WIDTH * POINT, rgb("#2c5d9e"), 1.0),
              Strokes(vertices[None], PATH_WIDTH * POINT, rgb("#c0392b"), 1.0, SOLID, Cap.PROJECTING),
              Marks(vertices[:1], Marker.DISC, 0.0, (start + OUTLINE / 2) * POINT, rgb("#e67e22"), 1.0),
              Marks(vertices[-1:], Marker.DISC, (last - OUTLINE / 2) * POINT, (last + OUTLINE / 2) * POINT,
                    rgb("#c0392b"), 1.0)]
    return planar.draw(window(BOX[:, 0], BOX[:, 1], ROWS), layers, np.ones(3))


def picture(outer: core.Conic, inner: core.Envelope, vertices: core.Point) -> np.ndarray:
    """For each inner conic, the ellipse, the inner conic and the path of sides about it from the same start, side by
    side: RGB bytes."""
    left, right = (panel(outer, inner[index], vertices[:, index]) for index in range(2))
    return np.concatenate([left, np.full((ROWS, GAP, 3), 255, dtype=np.uint8), right], axis=1)


def draw(outer: core.Conic, inner: core.Envelope, vertices: core.Point) -> plt.Figure:
    """For each inner conic, the ellipse, the inner conic and the path of sides about it from the same
    start, the start orange and the last vertex ringed."""
    return image_figure(picture(outer, inner, vertices))


def animate(scenes: list[tuple[core.Conic, core.Envelope, core.Point]]) -> Iterator[np.ndarray]:
    """One frame per scene."""
    for scene in scenes:
        yield picture(*scene)
