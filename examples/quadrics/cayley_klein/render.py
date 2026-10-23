"""Drawing for the Cayley-Klein example, on the GPU by `planar`: the level sets of polarities, the curves where they
vanish; lines, whole across the view; and points, as marks. The angles' labels over it in matplotlib."""

from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from examples.util.wgsl import pinned
from examples.util.wgsl import planar
from examples.util.wgsl.planar import Curves, Lines, Marker, Marks, window
from examples.quadrics.cayley_klein.core import Line, Point, Polarity, Scalar, ga

BOX = (-1.2, 1.9, -1.2, 1.6)
# The rows of pixels down the view.
ROWS = 680
# Pixels a point, at the figure's 120 dots per inch; the width of the marks' outlines.
POINT = 120 / 72
OUTLINE = 1.5


def euclidean(points: Point) -> np.ndarray:
    """Dehomogenize points to (..., 2) chart coordinates."""
    k = points.cast(ga.subspace("yw wx xy")).kernel
    return k[..., :2] / k[..., 2:]


def rgb(colour: str) -> np.ndarray:
    return np.array(mcolors.to_rgb(colour))


def marks(points: Point, colour: str, area: float, marker: Marker) -> Marks:
    """Points as marks of the area, in square points, with an outline in their colour."""
    return Marks(points.reshape(-1), marker, 0.0, (np.sqrt(area) / 2 + OUTLINE / 2) * POINT, rgb(colour), 1.0)


def level_sets(quadrics: Polarity, colour: str, width: float) -> Curves:
    """Each locus P & quadric(P) == 0, `width` points across."""
    return Curves(quadrics.reshape(-1), width * POINT, rgb(colour), 1.0)


def lines(lines: Line, colour: str, width: float) -> Lines:
    """Lines whole across the view, `width` points across."""
    return Lines(lines.reshape(-1), width * POINT, rgb(colour), 1.0)


def draw_hyperbolic_plane(
    C: Polarity, vertices: Point, sides: Line, angles: Scalar, area: Scalar, P: Point, foot: Point,
    normal: Line, reflected: Point, pole: Point, circles: Polarity, centres: Point,
) -> plt.Figure:
    """Left, inside the absolute, black: the triangle, blue, with its angles, the perpendicular, red, from
    the point, red, to its foot, dark red square, the point's reflection, orange, and the side's pole,
    purple diamond. Right, the circles about the two centres, blue and orange."""
    view = window(BOX[:2], BOX[2:], ROWS)
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 6), dpi=120)
    for ax in (left, right):
        ax.set_axis_off()
    fig.tight_layout()
    hyperbolic = [level_sets(C[None], "black", 2.0),
                  lines(sides, "#2563eb", 2.0),
                  lines(normal[None], "#dc2626", 1.6),
                  marks(vertices, "#1d4ed8", 40, Marker.DISC),
                  marks(P[None], "#dc2626", 50, Marker.DISC),
                  marks(foot[None], "#b91c1c", 40, Marker.SQUARE),
                  marks(reflected[None], "#f97316", 45, Marker.DISC),
                  marks(pole[None], "#a855f7", 45, Marker.DIAMOND)]
    pinned(left, planar.draw(view, hyperbolic, np.ones(3)), view.extent)
    for xy, theta in zip(euclidean(vertices), angles.to_array()):
        left.annotate(f"{np.degrees(theta):.1f}°", xy, textcoords="offset points", xytext=(6, 6), fontsize=9)

    rings = [level_sets(C[None], "black", 2.0)] + [level_sets(family, colour, 1.4)
                                                    for family, colour in zip(circles, ("#3b82f6", "#f97316"))]
    pinned(right, planar.draw(view, rings + [marks(centres, "#111827", 35, Marker.DISC)], np.ones(3)), view.extent)
    return fig
