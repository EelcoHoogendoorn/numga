"""Drawing for Poncelet's porism, on the GPU by `planar`: the two pairs of quadrics side by side, the ellipsoid the curve
where it vanishes and each inner quadric the curve its tangents touch, and the path of five sides from the same start
about each inner quadric."""

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
box = np.array([[-1.8, -1.9], [2.6, 1.5]])
rows = 232
gap = 9
# Pixels a point, at 63 dots per inch. The widths of the ellipsoid, of the inner quadric and of the path; the areas of
# the marks of the start and of the ring about the last vertex, in square points, and the width of their outlines.
pixels_per_point = 63 / 72
outer_width, inner_width, path_width = 1.6, 1.4, 1.4
start_area, last_area, outline = 50.0, 110.0, 1.5


def rgb(colour: str) -> np.ndarray:
    return np.array(mcolors.to_rgb(colour))


def panel(outer: core.Quadric, inner: core.Envelope, vertices: core.Point) -> np.ndarray:
    """The ellipsoid, an inner quadric and the path of sides about it, the start orange and the last vertex ringed: RGB
    bytes."""
    start, last = np.sqrt(start_area) / 2, np.sqrt(last_area) / 2
    layers = [Curves(outer[None], outer_width * pixels_per_point, rgb("0.25"), 1.0),
              Envelopes(inner[None], inner_width * pixels_per_point, rgb("#2c5d9e"), 1.0),
              Strokes(vertices[None], path_width * pixels_per_point, rgb("#c0392b"), 1.0, SOLID, Cap.PROJECTING),
              Marks(vertices[:1], Marker.DISC, 0.0, (start + outline / 2) * pixels_per_point, rgb("#e67e22"), 1.0),
              Marks(vertices[-1:], Marker.DISC, (last - outline / 2) * pixels_per_point,
                    (last + outline / 2) * pixels_per_point, rgb("#c0392b"), 1.0)]
    return planar.draw(window(box[:, 0], box[:, 1], rows), layers, np.ones(3))


def picture(outer: core.Quadric, inner: core.Envelope, vertices: core.Point) -> np.ndarray:
    """For each inner quadric, the ellipsoid, the inner quadric and the path of sides about it from the same start, side by
    side: RGB bytes."""
    left, right = (panel(outer, inner[index], vertices[:, index]) for index in range(2))
    return np.concatenate([left, np.full((rows, gap, 3), 255, dtype=np.uint8), right], axis=1)


def draw(outer: core.Quadric, inner: core.Envelope, vertices: core.Point) -> plt.Figure:
    """For each inner quadric, the ellipsoid, the inner quadric and the path of sides about it from the same
    start, the start orange and the last vertex ringed."""
    return image_figure(picture(outer, inner, vertices))


def animate(scenes: list[tuple[core.Quadric, core.Envelope, core.Point]]) -> Iterator[np.ndarray]:
    """One frame per scene."""
    for scene in scenes:
        yield picture(*scene)
