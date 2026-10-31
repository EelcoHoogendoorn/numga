"""Drawing for Poncelet's porism, on the GPU under two cameras of the plane: the two pairs of quadrics side by side, the
ellipsoid the curve where it vanishes, read by planes, and each inner quadric the curve its tangents touch, the curve of
its inverse read by lines; and the path of five sides from the same start about each inner quadric."""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from examples.util.animation import capture
from examples.util.wgsl import quadrics, simplices
from examples.util.wgsl.camera import Camera, flat_panel, panel
from examples.util.wgsl.simplices import Cap, Mesh
from examples.math.poncelet import core

mv = core.mv
# The view: x from -1.8 to 2.6, y from -1.9 to 1.5; the rows of pixels down it, and the columns between the two.
box = np.array([[-1.8, -1.9], [2.6, 1.5]])
rows = 232
gap = 9
# Pixels a point, at 63 dots per inch. The widths of the ellipsoid, of the inner quadric and of the path; the areas of
# the marks of the start and of the ring about the last vertex, in square points, and the width of their outlines.
pixels_per_point = 63 / 72
outer_width, inner_width, path_width = 1.6, 1.4, 1.4
start_area, last_area, outline = 50.0, 110.0, 1.5
# The radius of the start's mark, and of the ring about the last vertex, in pixels.
start_radius = (np.sqrt(start_area) / 2 + outline / 2) * pixels_per_point
last_radius = np.sqrt(last_area) / 2 * pixels_per_point


def rgb(colour: str) -> np.ndarray:
    return np.array(mcolors.to_rgb(colour))


def canvas() -> tuple[plt.Figure, list[tuple[plt.Axes, Camera, Camera]]]:
    """The figure, and for each of its two panels side by side its axes on the view and its two cameras: planes
    reading points, and lines, joins of points, reading poles, meets of planes, each pairing with what it reads."""
    columns = round(rows * np.ptp(box[:, 0]) / np.ptp(box[:, 1]))
    width = 2 * columns + gap
    figure = plt.figure(figsize=(width / 100, rows / 100), dpi=100)
    origin = mv.w.dual()                                                       # [] Point-
    panels = []
    for left in (0, columns + gap):
        ax = figure.add_axes((left / width, 0, columns / width, 1))
        ax.set(xlim=box[:, 0], ylim=box[:, 1])
        # The lines through the origin along y and along x, which read a pole's place across and up, and the line at
        # infinity, which reads its weight.
        lines = panel(ax, mv.y.dual() & origin, origin & mv.x.dual())._replace(weight=mv.x.dual() & mv.y.dual())
        panels.append((ax, flat_panel(ax, core.ga), lines))
    return figure, panels


def panel_pictures(planes: Camera, lines: Camera, outer: core.Quadric, inner: core.Envelope,
                   vertices: core.Point) -> list[np.ndarray]:
    """The ellipsoid, an inner quadric and the path of sides about it, the start orange and the last vertex ringed."""
    return [quadrics.curves(outer[None], outer_width * pixels_per_point, rgb("0.25"), planes),
            quadrics.curves(inner[None].inverse(), inner_width * pixels_per_point, rgb("#2c5d9e"), lines),
            simplices.strokes(vertices, Mesh.chains([len(vertices)]), rgb("#c0392b"), path_width * pixels_per_point,
                              planes, cap=Cap.PROJECTING),
            # The last vertex's ring, a white dot over a larger red one, under the start's dot, which it rings where
            # the path closes.
            simplices.markers(vertices[-1:], rgb("#c0392b"), last_radius + outline / 2 * pixels_per_point, planes),
            simplices.markers(vertices[-1:], np.ones(3), last_radius - outline / 2 * pixels_per_point, planes),
            simplices.markers(vertices[:1], rgb("#e67e22"), start_radius, planes)]


def draw(outer: core.Quadric, inner: core.Envelope, vertices: core.Point) -> plt.Figure:
    """For each inner quadric, the ellipsoid, the inner quadric and the path of sides about it from the same
    start, the start orange and the last vertex ringed."""
    figure, panels = canvas()
    for index, (ax, planes, lines) in enumerate(panels):
        simplices.overlay(ax, panel_pictures(planes, lines, outer, inner[index], vertices[:, index]))
    return figure


def animate(scenes: list[tuple[core.Quadric, core.Envelope, core.Point]]) -> Iterator[np.ndarray]:
    """One frame per scene."""
    figure, panels = canvas()
    outer, inner, vertices = scenes[0]
    images = [simplices.overlay(ax, panel_pictures(planes, lines, outer, inner[index], vertices[:, index]))
              for index, (ax, planes, lines) in enumerate(panels)]
    for outer, inner, vertices in scenes:
        for index, (image, (ax, planes, lines)) in enumerate(zip(images, panels)):
            image.set_data(simplices.composited(panel_pictures(planes, lines, outer, inner[index], vertices[:, index])))
        yield capture(figure)
    plt.close(figure)
