"""GPU rendering for Pascal's theorem under two planar cameras.

One camera pairs planes with points to draw the quadric curve, the hexagon edges, and the six corners; the other pairs point-join lines with line-meet crossings to draw the extended side lines, the Pascal line, and the three crossings.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from examples.math.pascal import core
from examples.util.animation import capture
from examples.util.panels import overlay, panel
from numga.render import Camera, drawing, quadrics, simplices
from numga.topology import Topology

mv = core.mv
# Viewport bounds in x and y, and vertical resolution in pixels.
box = np.array([[-2.5, -2.5], [4.5, 5.5]])
rows = 320
# Colours shared by each pair of opposite sides and their crossing.
pair_colours = np.array(["#1f77b4", "#2a9d4a", "#9b59b6"])
# Pixels per typographic point, stroke widths, and marker areas in square points.
point = 0.75
quadric_width, side_width, outline_width, pascal_width = 1.6, 1.1, 1.6, 2.0
corner_area, sixth_area, crossing_area, outline = 28.0, 60.0, 55.0, 1.5
corner_radius, sixth_radius = (np.sqrt(np.array([corner_area, sixth_area])) / 2 + outline / 2) * point
ring = np.sqrt(crossing_area) / 2 * point
hexagon_edges = Topology.loops([6])


def rgb(colours) -> np.ndarray:
    """Colour specifications convert to `[n_colours, 3]` float RGB triples."""
    return np.array([mcolors.to_rgb(colour) for colour in np.atleast_1d(colours)])


def canvas() -> tuple[plt.Figure, plt.Axes, Camera, Camera]:
    """Creates the viewport figure, its axes, a plane camera that pairs with points, and a line camera that pairs with crossings."""
    columns = round(rows * np.ptp(box[:, 0]) / np.ptp(box[:, 1]))
    figure = plt.figure(figsize=(columns / 100, rows / 100), dpi=100)
    ax = figure.add_axes((0, 0, 1, 1))
    ax.set(xlim=box[:, 0], ylim=box[:, 1])
    origin = mv.w.dual()                                                       # [] Point-
    # Coordinate lines through the origin and the line at infinity read a crossing's position and weight.
    lines = panel(ax, mv.y.dual() & origin, origin & mv.x.dual(), mv.x.dual() & mv.y.dual())
    return figure, ax, panel(ax, mv.x, mv.y, mv.w), lines


def pictures(planes: Camera, lines: Camera, shape: core.Quadric, hexagon: core.Point, sides: core.Line,
             crossings: core.Crossing, pascal_line: core.Line) -> list[np.ndarray]:
    """Renders the RGBA layers for the quadric, extended side lines, hexagon edges, Pascal line, corners, and ringed crossings."""
    return [quadrics.curves(shape[None], quadric_width * point, rgb("0.25"), planes),
            quadrics.flats(sides, side_width * point, rgb(np.tile(pair_colours, 2)), lines, 0.6),
            simplices.strokes(hexagon, hexagon_edges, rgb(np.tile(pair_colours, 2)), outline_width * point, planes),
            quadrics.flats(pascal_line[None], pascal_width * point, rgb("#c0392b"), lines),
            simplices.markers(hexagon, rgb("0.15"), corner_radius, planes),
            simplices.markers(hexagon[-1:], rgb("#e67e22"), sixth_radius, planes),
            # Draw each crossing in its pair's colour over a slightly larger red disc to form a ring.
            simplices.markers(crossings, rgb("#c0392b"), ring + outline / 2 * point, lines),
            simplices.markers(crossings, rgb(pair_colours), ring - outline / 2 * point, lines)]


def draw(shape: core.Quadric, hexagon: core.Point, sides: core.Line, crossings: core.Crossing,
         pascal_line: core.Line) -> plt.Figure:
    """Draws a single hexagon configuration on `shape` with its opposite-side crossings and Pascal line."""
    figure, ax, planes, lines = canvas()
    overlay(ax, pictures(planes, lines, shape, hexagon, sides, crossings, pascal_line))
    return figure


def animate(shape: core.Quadric,
            states: Iterable[tuple[core.Point, core.Line, core.Crossing, core.Line]]) -> Iterator[np.ndarray]:
    """Streams rendered RGB frames for `states` over the fixed quadric `shape`."""
    iterator = iter(states)
    first = next(iterator, None)
    if first is None:
        return
    figure, ax, planes, lines = canvas()
    overlay(ax, pictures(planes, lines, shape, *first))
    yield capture(figure)
    for state in iterator:
        ax.images[0].set_data(drawing.composited(pictures(planes, lines, shape, *state)))
        yield capture(figure)
    plt.close(figure)
