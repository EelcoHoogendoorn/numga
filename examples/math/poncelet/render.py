"""Side-by-side GPU rendering of Poncelet polygons between an outer point quadric and two inner line
envelopes.

Planes read points on the outer quadric, while point-join lines read poles of each inverted inner
envelope so both curves are drawn directly from their algebraic maps without coordinate extraction.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from examples.util.animation import capture
from examples.util.panels import overlay, panel
from numga.render import Camera, Cap, drawing, quadrics, simplices
from numga.topology import Topology
from examples.math.poncelet import core

mv = core.mv
# Viewport bounds across `x` and `y`, pixel row count, and inter-panel pixel gap.
view_box = np.array([[-1.8, -1.9], [2.6, 1.5]])                                # [2, 2]
n_rows = 232
panel_gap = 9
# Stroke widths and endpoint marker sizes converted from typographic points at 63 pixels per inch.
pixels_per_point = 63 / 72
outer_width, inner_width, path_width = 1.6, 1.4, 1.4
start_area, last_area, outline_width = 50.0, 110.0, 1.5
start_radius = (np.sqrt(start_area) / 2 + outline_width / 2) * pixels_per_point
last_radius = np.sqrt(last_area) / 2 * pixels_per_point


def rgb(colour: str) -> np.ndarray:
    """A matplotlib colour string maps to its `[3]` float RGB triple."""
    return np.array(mcolors.to_rgb(colour))


def canvas() -> tuple[plt.Figure, list[tuple[plt.Axes, Camera, Camera]]]:
    """Two side-by-side axes pair a plane camera for points with a line camera for poles across
    `view_box`."""
    n_columns = round(n_rows * np.ptp(view_box[:, 0]) / np.ptp(view_box[:, 1]))
    total_width = 2 * n_columns + panel_gap
    figure = plt.figure(figsize=(total_width / 100, n_rows / 100), dpi=100)
    origin = mv.w.dual()                                                       # [] Point-
    panels = []
    for left in (0, n_columns + panel_gap):
        ax = figure.add_axes((left / total_width, 0, n_columns / total_width, 1))
        ax.set(xlim=view_box[:, 0], ylim=view_box[:, 1])
        # Coordinate lines through the origin and the horizon line read a pole's coordinates and weight by pairing.
        lines = panel(ax, mv.y.dual() & origin, origin & mv.x.dual(), mv.x.dual() & mv.y.dual())
        panels.append((ax, panel(ax, mv.x, mv.y, mv.w), lines))
    return figure, panels


def quadric_layers(planes: Camera, lines: Camera, outer: core.Quadric,
                   inner: core.Envelope) -> list[np.ndarray]:
    """Implicit GPU shaders rasterize `outer` under `planes` and the inverse of `inner` under
    `lines`."""
    return [
        quadrics.curves(outer[None], outer_width * pixels_per_point, rgb("0.25"), planes),
        quadrics.curves(inner[None].inverse(), inner_width * pixels_per_point, rgb("#2c5d9e"), lines),
    ]


def path_layers(planes: Camera, vertices: core.Point) -> list[np.ndarray]:
    """Stroke and marker shaders rasterize the polygonal path, the ringed final vertex, and the
    orange start."""
    n_vertices = len(vertices)
    return [
        simplices.strokes(vertices, Topology.chains([n_vertices]), rgb("#c0392b"),
                          path_width * pixels_per_point, planes, cap=Cap.PROJECTING),
        # Layering a white disc inside a red disc under the orange start dot leaves a visible ring when the path closes.
        simplices.markers(vertices[-1:], rgb("#c0392b"), last_radius + outline_width / 2 * pixels_per_point, planes),
        simplices.markers(vertices[-1:], np.ones(3), last_radius - outline_width / 2 * pixels_per_point, planes),
        simplices.markers(vertices[:1], rgb("#e67e22"), start_radius, planes),
    ]


def paths_figure(outer: core.Quadric, inner: core.Envelope, vertices: core.Point) -> plt.Figure:
    """Both panels composite the quadric curves and polygonal paths for a single
    `[n_sides + 1, 2] Point-` state."""
    figure, panels = canvas()
    for index, (ax, planes, lines) in enumerate(panels):
        layers = quadric_layers(planes, lines, outer, inner[index]) + path_layers(planes, vertices[:, index])
        overlay(ax, layers)
    return figure


def path_frames(outer: core.Quadric, inner: core.Envelope, paths: Iterable[core.Point]) -> Iterator[np.ndarray]:
    """Yields an antialiased RGB frame for each `[n_sides + 1, 2] Point-` path pair in `paths`."""
    figure, panels = canvas()
    backgrounds = [
        quadric_layers(planes, lines, outer, inner[index])
        for index, (_, planes, lines) in enumerate(panels)
    ]
    images = None
    try:
        for vertices in paths:
            layers_per_panel = [
                backgrounds[index] + path_layers(planes, vertices[:, index])
                for index, (_, planes, _) in enumerate(panels)
            ]
            if images is None:
                images = [overlay(ax, layers) for (ax, _, _), layers in zip(panels, layers_per_panel)]
            else:
                for image, layers in zip(images, layers_per_panel):
                    image.set_data(drawing.composited(layers))
            yield capture(figure)
    finally:
        plt.close(figure)
