"""The two Fresnel sheets and their meridian sections."""

from __future__ import annotations

from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from matplotlib.colors import to_rgb

from examples.util.animation import capture
from examples.electromagnetism.fresnel import core
from examples.util.wgsl import labels, meshes, show, solids, vec3

sheet_colours = ("#2563eb", "#ea580c")
view_limit = 2.05
latitude_stride = 4
longitude_stride = 8
sm = solids.mv
# How much of what lies behind them each side of a sheet and its grid lines hide, and the axes' colour.
sheet_opacity, grid_opacity = 0.12, 0.4
axis_colour = np.array([0.45, 0.47, 0.52])
# The view's elevation and azimuth in radians, azimuth zero looking along +y; the half width of the square view about
# the origin, the same for every medium so their sizes compare; and the pixels across it.
elevation, azimuth = np.radians(22.0), np.radians(37.0)
half_width, n_pixels = 2.3, 400
# The widths of the grid lines and the axes' arrows, and the arrowheads' length and width, in pixels; how far the
# axes reach and their names lie, in units of k.
grid_width, arrow_width, head_length, head_width = 2.0, 2.0, 10.0, 7.0
axis_length, label_distance = 1.95, 2.12


# --- plumbing -------------------------------------------------------------------------
def sheet_faces(n_latitudes: int, n_longitudes: int) -> np.ndarray:
    """Triangles over a grid of points from pole to pole, `[latitudes, longitudes]`: two to each cell, but one in the
    cells at either pole, whose two corners on the pole coincide."""
    index = np.arange(n_latitudes * n_longitudes).reshape(n_latitudes, n_longitudes)
    corners = np.stack([index[:-1, :-1], index[1:, :-1], index[1:, 1:], index[:-1, 1:]], axis=-1)
    return np.concatenate([corners[:-1, :, [0, 1, 2]].reshape(-1, 3), corners[1:, :, [0, 2, 3]].reshape(-1, 3)])


def gridded(colour: np.ndarray, n_latitudes: int, near: bool) -> str:
    """WGSL colouring one side of a sheet, the side the camera sees the front of or the far side, translucent, inked
    more opaquely along every few parallels short of the poles and every few meridians: by how many pixels the
    corners' grid indices, `values.x` and `values.y`, lie from the nearest line, their change across a pixel the
    measure. The other side hides nothing, and is left for the other mesh."""
    return f"""
fn colour(values: vec4<f32>, face_values: vec4<f32>, corner: vec3<f32>, front: bool) -> vec4<f32> {{
    let strides = vec2<f32>({latitude_stride:.1f}, {longitude_stride:.1f});
    let nearest = round(values.xy / strides) * strides;
    let steps = oversampling * fwidth(values.xy);
    var ink = clamp({grid_width / 2 + 0.5} - abs(values.xy - nearest) / steps, vec2<f32>(0.0), vec2<f32>(1.0));
    // No parallel on the poles, where they shrink to points.
    ink.x *= select(0.0, 1.0, nearest.x > 0 && nearest.x < {n_latitudes - 1:.1f});
    let shown = select(0.0, 1.0, front == {str(near).lower()});
    return vec4<f32>({vec3(colour)}, shown * mix({sheet_opacity}, {grid_opacity}, max(ink.x, ink.y)));
}}
"""


def k_axes(view: solids.View) -> list[tuple[solids.Quadric, np.ndarray]]:
    """The frame of k-space, its shafts and its heads."""
    shafts, heads = solids.frame(axis_length, arrow_width / 2 * view.pixel, head_length * view.pixel,
                                 head_width / 2 * view.pixel)
    return [(shafts, axis_colour), (heads, axis_colour)]


def surface(ax: Axes, sheets: core.Vector) -> None:
    """Both sheets, translucent, with grid lines along every few latitudes and longitudes, in the frame of k-space,
    its axes named at their tips."""
    points = sheets.cast(core.ga.subspace("x y z")).kernel                       # [latitudes, longitudes, 2, 3]
    view = solids.View(solids.aim(solids.origin, elevation, azimuth), np.full(2, half_width), n_pixels)
    n_latitudes, n_longitudes = points.shape[:2]
    faces = sheet_faces(n_latitudes, n_longitudes)
    # Each vertex's place on the grid, latitude and longitude, for the grid lines inked on the sheet.
    indices = np.stack(np.meshgrid(np.arange(n_latitudes), np.arange(n_longitudes), indexing="ij"), axis=-1)
    grid = indices.reshape(-1, 2)[faces]                                         # [faces, 3, 2]
    # Each sheet's near side and far side a mesh of its own, so that the far side shows through the near.
    sides = []
    for sheet, colour in zip(np.moveaxis(points, 2, 0), map(to_rgb, sheet_colours)):
        places = sm.point(sheet.reshape(-1, 3))
        normals = meshes.smooth(places, faces)
        sides += [(meshes.samples(places, faces, grid, normals, view, colour=gridded(np.array(colour), n_latitudes, near),
                                   gloss=0.2), 1.0) for near in (True, False)]
    show(ax, solids.draw(k_axes(view), view, np.ones(3), meshes=sides), view)
    labels(ax, view, sm.point(label_distance * np.eye(3)), ["$k_x$", "$k_y$", "$k_z$"], ha="center", va="center",
           fontsize=11, color=axis_colour)


def section(ax: Axes, sections: core.Vector) -> None:
    points = sections.cast(core.ga.subspace("x y z")).kernel
    for mode, (colour, style, label) in enumerate(zip(sheet_colours, ("-", "--"), ("inner sheet", "outer sheet"))):
        ax.plot(points[:, mode, 0], points[:, mode, 2], color=colour, linestyle=style, linewidth=2, label=label)
    ax.axhline(0, color="#cbd5e1", linewidth=0.7)
    ax.axvline(0, color="#cbd5e1", linewidth=0.7)
    # The axes named at their ends, without scales: the sheets' shapes against each other are the point.
    ax.text(view_limit, 0, "$k_x$", ha="right", va="bottom", fontsize=11, color=axis_colour)
    ax.text(0, view_limit, "$k_z$", ha="left", va="top", fontsize=11, color=axis_colour)
    ax.set(xlim=(-view_limit, view_limit), ylim=(-view_limit, view_limit))
    ax.set_aspect("equal")
    ax.set_axis_off()


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
