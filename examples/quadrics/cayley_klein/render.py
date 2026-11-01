"""Drawing for the Cayley-Klein example, on the GPU under two cameras of the plane: the level sets of polarities, the
curves where they vanish, and planes, whole across the view, read by planes; lines, whole across the view, read by
lines, joins of points; and points, as marks. The angles' labels over it in matplotlib."""

from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from examples.util import charts
from examples.util.wgsl import quadrics, simplices
from examples.util.wgsl.camera import Camera, flat_panel, panel, read
from numga import stack
from examples.quadrics.cayley_klein.core import Line, Plane, Point, Polarity, Scalar, ga, mv

BOX = (-1.2, 1.9, -1.2, 1.6)
# The pixels of a point the marks' areas and the strokes' widths below are given in, a point at 120 dots per inch;
# the width of the marks' outlines, and of the strokes of the square and the diamond.
POINT = 120 / 72
OUTLINE = 1.0
FRAME = 1.2


def euclidean(points: Point) -> np.ndarray:
    """Chart coordinates of points, `[..., 2]`: their pairings with the planes x = 0 and y = 0, over their weights."""
    return ((stack((mv.x, mv.y)) & points[..., None]) / (mv.w & points)[..., None]).to_array()


def rgb(colour: str) -> np.ndarray:
    return np.array(mcolors.to_rgb(colour))


def cameras(ax: plt.Axes) -> tuple[Camera, Camera]:
    """The panel on the view and its two cameras: planes reading points, and lines, joins of points, reading meets of
    planes, each pairing with what it reads."""
    ax.set(xlim=BOX[:2], ylim=BOX[2:])
    origin = mv.w.dual()                                                       # [] Point-
    # The lines through the origin along y and along x, which read a meet's place across and up, and the line at
    # infinity, which reads its weight.
    lines = panel(ax, mv.y.dual() & origin, origin & mv.x.dual())._replace(weight=mv.x.dual() & mv.y.dual())
    return flat_panel(ax, ga), lines


def marks(points: Point, colour: str, area: float, camera: Camera) -> np.ndarray:
    """Points as dots of the area, in square points, with an outline in their colour."""
    return simplices.markers(points.reshape(-1), rgb(colour), (np.sqrt(area) / 2 + OUTLINE / 2) * POINT, camera)


def squares(points: Point, colour: str, area: float, turn: float, camera: Camera) -> np.ndarray:
    """Points as squares of the area, in square points, each four strokes in their colour, the first corner turned
    `turn` from the direction the camera reads as one across: an eighth of a turn for a square along the picture's
    rows and columns, none for a diamond."""
    # The direction read as one across, turned to each corner and round to the first again, half a diagonal long.
    along = camera.across.dual() / read(camera.across, camera.across.dual())   # [] Point-
    turns = (mv.xy * (-(turn + np.pi / 2 * np.arange(5)) / 2)).exp()        # [corners + 1] Rotor+
    corners = (turns >> along) * (np.sqrt(2 * area) / 2 * POINT * camera.pixel)   # [corners + 1] Point-
    places = (points / (mv.w & points))[:, None] + corners                   # [points, corners + 1] Point-
    return simplices.strokes(places.reshape(-1), simplices.Topology.chains([5] * len(points)), rgb(colour),
                             FRAME * POINT, camera)


def level_sets(polarities: Polarity, colour: str, width: float, camera: Camera) -> np.ndarray:
    """Each locus P & polarity(P) == 0, `width` points across."""
    return quadrics.curves(polarities.reshape(-1), width * POINT, rgb(colour), camera)


@charts.style()
def draw_hyperbolic_space(
    absolute: Polarity, vertices: Point, sides: Plane, angles: Scalar, probe: Point, foot: Point,
    normal: Line, reflected: Point, pole: Point, spheres: Polarity, centres: Point,
) -> plt.Figure:
    """Left, inside the absolute, black: the triangle, blue, with its angles, the perpendicular, red, from
    the point, red, to its foot, dark red square, the point's reflection, orange, and the side's pole,
    purple diamond. Right, the spheres about the two centres, blue and orange."""
    # Two panels, each the room of its picture, and the angles' labels over the first.
    fig, (left, right) = charts.subplots(780, 360, ncols=2)
    for ax in (left, right):
        ax.set_axis_off()
    fig.tight_layout()
    planes, lines = cameras(left)
    simplices.overlay(left, [level_sets(absolute[None], "black", 1.2, planes),
                             quadrics.flats(sides, 1.2 * POINT, rgb("#2563eb"), planes),
                             quadrics.flats(normal[None], 1.0 * POINT, rgb("#dc2626"), lines),
                             marks(vertices, "#1d4ed8", 14, planes),
                             marks(probe[None], "#dc2626", 18, planes),
                             squares(foot[None], "#b91c1c", 14, np.pi / 4, planes),
                             marks(reflected[None], "#f97316", 16, planes),
                             squares(pole[None], "#a855f7", 16, 0.0, planes)])
    for xy, theta in zip(euclidean(vertices), angles.to_array()):
        left.annotate(f"{np.degrees(theta):.1f}°", xy, textcoords="offset points", xytext=(3, 3))

    planes, _ = cameras(right)
    simplices.overlay(right, [level_sets(absolute[None], "black", 1.2, planes)]
                      + [level_sets(family, colour, 0.8, planes)
                         for family, colour in zip(spheres, ("#3b82f6", "#f97316"))]
                      + [marks(centres, "#111827", 13, planes)])
    return fig
