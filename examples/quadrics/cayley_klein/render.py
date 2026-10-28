"""Drawing for the Cayley-Klein example, on the GPU by `planar`: the level sets of polarities, the curves where they
vanish; planes and lines, whole across the view; and points, as marks. The angles' labels over it in matplotlib."""

from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from examples.util.wgsl import pinned
from examples.util.wgsl import planar
from examples.util.wgsl.planar import Curves, Flats, Marker, Marks, window
from numga import stack
from examples.quadrics.cayley_klein.core import Line, Plane, Point, Polarity, Scalar, mv

BOX = (-1.2, 1.9, -1.2, 1.6)
# The rows of pixels down the view.
ROWS = 340
# Pixels a point, at the figure's 120 dots per inch; the width of the marks' outlines.
POINT = 120 / 72
OUTLINE = 1.0


def euclidean(points: Point) -> np.ndarray:
    """Chart coordinates of points, `[..., 2]`: their pairings with the planes x = 0 and y = 0, over their weights."""
    return ((stack((mv.x, mv.y)) & points[..., None]) / (mv.w & points)[..., None]).to_array()


def rgb(colour: str) -> np.ndarray:
    return np.array(mcolors.to_rgb(colour))


def marks(points: Point, colour: str, area: float, marker: Marker) -> Marks:
    """Points as marks of the area, in square points, with an outline in their colour."""
    return Marks(points.reshape(-1), marker, 0.0, (np.sqrt(area) / 2 + OUTLINE / 2) * POINT, rgb(colour), 1.0)


def level_sets(quadrics: Polarity, colour: str, width: float) -> Curves:
    """Each locus P & quadric(P) == 0, `width` points across."""
    return Curves(quadrics.reshape(-1), width * POINT, rgb(colour), 1.0)


def flats(flats: Plane, colour: str, width: float) -> Flats:
    """Planes and lines whole across the view, `width` points across."""
    return Flats(flats.reshape(-1), width * POINT, rgb(colour), 1.0)


def draw_hyperbolic_space(
    absolute: Polarity, vertices: Point, sides: Plane, angles: Scalar, probe: Point, foot: Point,
    normal: Line, reflected: Point, pole: Point, spheres: Polarity, centres: Point,
) -> plt.Figure:
    """Left, inside the absolute, black: the triangle, blue, with its angles, the perpendicular, red, from
    the point, red, to its foot, dark red square, the point's reflection, orange, and the side's pole,
    purple diamond. Right, the spheres about the two centres, blue and orange."""
    view = window(BOX[:2], BOX[2:], ROWS)
    fig, (left, right) = plt.subplots(1, 2, figsize=(6.5, 3), dpi=120)
    for ax in (left, right):
        ax.set_axis_off()
    fig.tight_layout()
    hyperbolic = [level_sets(absolute[None], "black", 1.2),
                  flats(sides, "#2563eb", 1.2),
                  flats(normal[None], "#dc2626", 1.0),
                  marks(vertices, "#1d4ed8", 14, Marker.DISC),
                  marks(probe[None], "#dc2626", 18, Marker.DISC),
                  marks(foot[None], "#b91c1c", 14, Marker.SQUARE),
                  marks(reflected[None], "#f97316", 16, Marker.DISC),
                  marks(pole[None], "#a855f7", 16, Marker.DIAMOND)]
    pinned(left, planar.draw(view, hyperbolic, np.ones(3)), view.extent)
    for xy, theta in zip(euclidean(vertices), angles.to_array()):
        left.annotate(f"{np.degrees(theta):.1f}°", xy, textcoords="offset points", xytext=(3, 3), fontsize=7)

    rings = [level_sets(absolute[None], "black", 1.2)] + [level_sets(family, colour, 0.8)
                                                    for family, colour in zip(spheres, ("#3b82f6", "#f97316"))]
    pinned(right, planar.draw(view, rings + [marks(centres, "#111827", 13, Marker.DISC)], np.ones(3)), view.extent)
    return fig
