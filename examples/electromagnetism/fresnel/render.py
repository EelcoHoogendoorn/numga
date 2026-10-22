"""The two Fresnel sheets and their meridian sections."""

from __future__ import annotations

from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from matplotlib.colors import to_rgb

from examples.animation import capture
from examples.electromagnetism.fresnel import core
from examples.wgsl import behind, compose, labels, meshes, show, solids

sheet_colours = ("#2563eb", "#ea580c")
view_limit = 2.05
latitude_stride = 4
longitude_stride = 8
sm = solids.mv
# The sheets' opacity and their grid lines', and the axes' colour.
sheet_opacity, grid_opacity = 0.2, 0.55
axis_colour = np.array([0.45, 0.47, 0.52])
# The view's elevation and azimuth in radians, azimuth zero looking along +y; the half width of the square view about
# the origin, the same for every medium so their sizes compare; and the pixels across it.
elevation, azimuth = np.radians(22.0), np.radians(37.0)
half_width, n_pixels = 2.3, 400
# The widths of the grid lines and the axes' arrows, the arrowheads' length and width and the ticks' radius, in pixels;
# how far the axes reach and their names lie, and the ticks, in units of k.
grid_width, arrow_width, head_length, head_width, tick_radius = 3.0, 3.0, 12.0, 7.0, 3.5
axis_length, label_distance, ticks = 1.95, 2.12, np.array([-1.0, 1.0])


# --- plumbing -------------------------------------------------------------------------
def sheet_faces(n_latitudes: int, n_longitudes: int) -> np.ndarray:
    """Triangles over a grid of points from pole to pole, `[latitudes, longitudes]`: two to each cell, but one in the
    cells at either pole, whose two corners on the pole coincide."""
    index = np.arange(n_latitudes * n_longitudes).reshape(n_latitudes, n_longitudes)
    corners = np.stack([index[:-1, :-1], index[1:, :-1], index[1:, 1:], index[:-1, 1:]], axis=-1)
    return np.concatenate([corners[:-1, :, [0, 1, 2]].reshape(-1, 3), corners[1:, :, [0, 2, 3]].reshape(-1, 3)])


def k_axes(view: solids.View) -> tuple[np.ndarray, np.ndarray]:
    """The axes of k-space as arrows through the origin, with ticks at -1 and 1."""
    shafts, heads = solids.arrows(sm.point(-axis_length * np.eye(3)), sm.point(axis_length * np.eye(3)),
                                  arrow_width / 2 * view.pixel, head_length * view.pixel, head_width / 2 * view.pixel)
    marks = solids.ball(sm.point((ticks[:, None, None] * np.eye(3)).reshape(-1, 3)), tick_radius * view.pixel)
    return solids.layer([(shafts, axis_colour), (heads, axis_colour), (marks[:, None], axis_colour)], view)


def surface(ax: Axes, sheets: core.Vector) -> None:
    """Both sheets, translucent, with grid lines along every few latitudes and longitudes, about the axes of k-space,
    named at their tips."""
    points = sheets.cast(core.ga.subspace("x y z")).kernel                       # [latitudes, longitudes, 2, 3]
    view = solids.View(solids.aim(solids.origin, elevation, azimuth), np.full(2, half_width), n_pixels)
    faces = sheet_faces(*points.shape[:2])
    layers, lines = [], []
    for sheet, colour in zip(np.moveaxis(points, 2, 0), map(to_rgb, sheet_colours)):
        places = sm.point(sheet.reshape(-1, 3))
        # Pushed back a little, so the grid lines drawn on the sheet stay in front of it.
        layer = meshes.layer(places, faces, np.array(colour), meshes.smooth(places, faces), view, gloss=0.2)
        layers.append((*behind(layer, 2 * view.pixel), sheet_opacity))
        # Parallels short of the poles, where they shrink to points, and meridians short of the closing one.
        lines += [(line, colour) for line in sheet[latitude_stride:-1:latitude_stride]]
        lines += [(line, colour) for line in np.moveaxis(sheet[:, :-1:longitude_stride], 1, 0)]
    grid = meshes.tubes(lines, grid_width, view)
    show(ax, compose([*layers, (*grid, grid_opacity), (*k_axes(view), 1.0)], np.ones(3)), view)
    labels(ax, view, sm.point(label_distance * np.eye(3)), ["$k_x$", "$k_y$", "$k_z$"], ha="center", va="center",
           fontsize=11, color=axis_colour)


def section(ax: Axes, sections: core.Vector) -> None:
    points = sections.cast(core.ga.subspace("x y z")).kernel
    for mode, (colour, style, label) in enumerate(zip(sheet_colours, ("-", "--"), ("inner sheet", "outer sheet"))):
        ax.plot(points[:, mode, 0], points[:, mode, 2], color=colour, linestyle=style, linewidth=2, label=label)
    ax.axhline(0, color="#cbd5e1", linewidth=0.7)
    ax.axvline(0, color="#cbd5e1", linewidth=0.7)
    ax.set(xlim=(-view_limit, view_limit), ylim=(-view_limit, view_limit), xlabel="$k_x$", ylabel="$k_z$")
    ax.set_aspect("equal")
    ax.spines[["top", "right"]].set_visible(False)


def comparison(sheets: core.Vector, sections: core.Vector, names: tuple[str, ...]) -> plt.Figure:
    figure = plt.figure(figsize=(12, 7.8), layout="constrained")
    count = len(names)
    for i, name in enumerate(names):
        ax = figure.add_subplot(2, count, i + 1)
        surface(ax, sheets[i])
        ax.set_title(name, fontsize=11)
        ax = figure.add_subplot(2, count, count + i + 1)
        section(ax, sections[i])
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    return figure


def frame(sheets: core.Vector, sections: core.Vector) -> np.ndarray:
    figure = plt.figure(figsize=(9, 4.5), dpi=85, layout="constrained")
    surface(figure.add_subplot(1, 2, 1), sheets)
    section(figure.add_subplot(1, 2, 2), sections)
    pixels = capture(figure)
    plt.close(figure)
    return pixels


def animate(frames: Iterable[tuple[core.Vector, core.Vector]]) -> list[np.ndarray]:
    return [frame(sheets, sections) for sheets, sections in frames]
