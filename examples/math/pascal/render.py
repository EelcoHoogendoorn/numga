"""Drawing for Pascal's theorem, on the GPU under two cameras of the plane: the quadric, the curve where it vanishes,
read by planes; the sides of the hexagon and the line through the crossings of opposite sides, whole across the view,
and the crossings as discs, read by lines; the hexagon's outline, and its corners as discs, read by planes."""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from examples.util.animation import capture
from examples.util.wgsl import quadrics, simplices
from examples.util.wgsl.camera import Camera, flat_panel, panel
from examples.util.wgsl.simplices import Topology
from examples.math.pascal import core

mv = core.mv
# The view: x from -2.5 to 4.5, y from -2.5 to 5.5, and the rows of pixels down it.
box = np.array([[-2.5, -2.5], [4.5, 5.5]])
rows = 320
# Colours of the three pairs of opposite sides and their crossings.
pair_colours = np.array(["#1f77b4", "#2a9d4a", "#9b59b6"])
# Pixels a point. The widths of the quadric, of the sides, of the hexagon's outline and of the line through the
# crossings; the areas of the marks of the corners, of the sixth corner and of the crossings, in square points, and the
# width of their outlines.
point = 0.75
quadric_width, side_width, outline_width, pascal_width = 1.6, 1.1, 1.6, 2.0
corner_area, sixth_area, crossing_area, outline = 28.0, 60.0, 55.0, 1.5
# The radii of the marks in pixels, a mark of an area with its outline about it; and of the crossings' rings.
corner_radius, sixth_radius = ((np.sqrt(np.array([corner_area, sixth_area])) / 2 + outline / 2) * point)
ring = np.sqrt(crossing_area) / 2 * point
# The hexagon's outline, each corner to the next.
hexagon_edges = Topology.loops([6])


def rgb(colours) -> np.ndarray:
    return np.array([mcolors.to_rgb(colour) for colour in np.atleast_1d(colours)])


def canvas() -> tuple[plt.Figure, plt.Axes, Camera, Camera]:
    """The figure, its axes on the view, and its two cameras: planes reading points, and lines, joins of points,
    reading crossings, meets of lines, each pairing with what it reads."""
    columns = round(rows * np.ptp(box[:, 0]) / np.ptp(box[:, 1]))
    figure = plt.figure(figsize=(columns / 100, rows / 100), dpi=100)
    ax = figure.add_axes((0, 0, 1, 1))
    ax.set(xlim=box[:, 0], ylim=box[:, 1])
    origin = mv.w.dual()                                                       # [] Point-
    # The lines through the origin along y and along x, which read a crossing's place across and up, and the line at
    # infinity, which reads its weight.
    lines = panel(ax, mv.y.dual() & origin, origin & mv.x.dual())._replace(weight=mv.x.dual() & mv.y.dual())
    return figure, ax, flat_panel(ax, core.ga), lines


def pictures(planes: Camera, lines: Camera, shape: core.Quadric, hexagon: core.Point, sides: core.Line,
             crossing: core.Crossing, pascal_line: core.Line) -> list[np.ndarray]:
    """The hexagon's outline under the quadric, its sides in the colours of their pairs and the line through the
    crossings; its corners, the sixth orange, and the crossings, ringed in the line's colour."""
    return [quadrics.curves(shape[None], quadric_width * point, rgb("0.25"), planes),
            quadrics.flats(sides, side_width * point, rgb(np.tile(pair_colours, 2)), lines, 0.6),
            simplices.strokes(hexagon, hexagon_edges, rgb(np.tile(pair_colours, 2)), outline_width * point, planes),
            quadrics.flats(pascal_line[None], pascal_width * point, rgb("#c0392b"), lines),
            simplices.markers(hexagon, rgb("0.15"), corner_radius, planes),
            simplices.markers(hexagon[-1:], rgb("#e67e22"), sixth_radius, planes),
            # Each crossing in its pair's colour, ringed in the line's: a dot over a larger dot.
            simplices.markers(crossing, rgb("#c0392b"), ring + outline / 2 * point, lines),
            simplices.markers(crossing, rgb(pair_colours), ring - outline / 2 * point, lines)]


def draw(shape: core.Quadric, hexagon: core.Point, sides: core.Line, crossing: core.Crossing,
         pascal_line: core.Line) -> plt.Figure:
    """Six points on a quadric as a hexagon, its opposite sides in matching colours, and the three
    points where they meet on one line."""
    figure, ax, planes, lines = canvas()
    simplices.overlay(ax, pictures(planes, lines, shape, hexagon, sides, crossing, pascal_line))
    return figure


def animate(scenes: list[tuple[core.Quadric, core.Point, core.Line, core.Crossing, core.Line]]) -> Iterator[np.ndarray]:
    """One frame per scene."""
    figure, ax, planes, lines = canvas()
    image = simplices.overlay(ax, pictures(planes, lines, *scenes[0]))
    for scene in scenes:
        image.set_data(simplices.composited(pictures(planes, lines, *scene)))
        yield capture(figure)
    plt.close(figure)
