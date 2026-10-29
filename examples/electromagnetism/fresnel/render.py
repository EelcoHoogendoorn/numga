"""The two Fresnel sheets and their meridian sections."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D

from matplotlib.colors import to_rgb

from numga import stack

from examples.util import charts
from examples.util.animation import capture
from examples.electromagnetism.fresnel import core
from examples.util.wgsl import fitted, labels, meshes, show, solids
from examples.util.wgsl.scene import Scene

sheet_colours = ("#2563eb", "#ea580c")
view_limit = 2.4
latitude_stride = 4
longitude_stride = 8
sm = solids.mv
# How much of what lies behind them each side of a sheet and its grid lines hide, and the axes' colour.
sheet_opacity, grid_opacity = 0.12, 0.4
axis_colour = np.array([0.45, 0.47, 0.52])
# The view's elevation and azimuth in radians, azimuth zero looking along +y, and the half width of the square view
# about the origin, the same for every medium so their sizes compare.
elevation, azimuth = np.radians(22.0), np.radians(37.0)
half_width = 2.3
# The widths of the grid lines and the axes' arrows, and the arrowheads' length and width, in pixels; how far the
# axes reach and their names lie, in units of k.
grid_width, arrow_width, head_length, head_width = 1.0, 1.0, 5.0, 3.5
axis_length, label_distance = 1.95, 2.12
# The figures' sizes in pixels: the comparison, each medium's sheets above their sections, three quarters of the
# column; the animation, the sheets beside their sections, about two thirds of it.
comparison_size, animation_size = (600, 400), (520, 260)


# --- plumbing -------------------------------------------------------------------------
def coordinates(wavevectors: core.Vector) -> np.ndarray:
    """The coordinates of spatial wavevectors, read along the axes of space `[..., 3]`."""
    mv = wavevectors.context.multivector
    return (wavevectors[..., None] & stack((mv.x, mv.y, mv.z)).dual()).to_array()


def gpu_points(wavevectors: core.Vector) -> solids.Point:
    """The renderer's points at the tips of spatial wavevectors in k-space: the one hand-off from the spacetime
    algebra to the renderer's."""
    return sm.point(coordinates(wavevectors))


def sheet_faces(n_latitudes: int, n_longitudes: int) -> np.ndarray:
    """Triangles over a grid of points from pole to pole, `[latitudes, longitudes]`: two to each cell, but one in the
    cells at either pole, whose two corners on the pole coincide."""
    index = np.arange(n_latitudes * n_longitudes).reshape(n_latitudes, n_longitudes)
    corners = np.stack([index[:-1, :-1], index[1:, :-1], index[1:, 1:], index[:-1, 1:]], axis=-1)
    return np.concatenate([corners[:-1, :, [0, 1, 2]].reshape(-1, 3), corners[1:, :, [0, 2, 3]].reshape(-1, 3)])


# WGSL colouring a sheet, translucent, inked more opaquely along every few parallels short of the poles and every few
# meridians: by how many pixels the corners' grid indices, `values.x` and `values.y`, lie from the nearest line, their
# change across a pixel the measure; and showing as much of each side, the side the camera sees the front of and the
# far side, as the constants `front_shown` and `back_shown` say.
gridded = """
fn colour(values: vec4<f32>, face_values: vec4<f32>, corner: vec3<f32>, front: bool) -> vec4<f32> {
    let nearest = round(values.xy / strides) * strides;
    let steps = oversampling * fwidth(values.xy);
    var ink = clamp(grid_width / 2 + 0.5 - abs(values.xy - nearest) / steps, vec2<f32>(0.0), vec2<f32>(1.0));
    // No parallel on the poles, where they shrink to points.
    ink.x *= select(0.0, 1.0, nearest.x > 0 && nearest.x < f32(n_latitudes - 1));
    let shown = select(back_shown, front_shown, front);
    return vec4<f32>(sheet_colour, shown * mix(sheet_opacity, grid_opacity, max(ink.x, ink.y)));
}
"""


def side_colour(colour: np.ndarray, n_latitudes: int, near: bool) -> meshes.Colour:
    """One side of a sheet in its colour, gridded, the side the camera sees the front of or the far side; the other
    side hides nothing, and is left for the other mesh."""
    return meshes.Colour(gridded, dict(sheet_colour=colour, strides=np.array([latitude_stride, longitude_stride]),
                                       grid_width=grid_width, n_latitudes=n_latitudes, sheet_opacity=sheet_opacity,
                                       grid_opacity=grid_opacity, front_shown=float(near), back_shown=float(not near)))


def k_axes(view: solids.View) -> list[tuple[solids.Quadric, np.ndarray]]:
    """The frame of k-space, its shafts and its heads."""
    shafts, heads = solids.frame(axis_length, arrow_width / 2 * view.pixel, head_length * view.pixel,
                                 head_width / 2 * view.pixel)
    return [(shafts, axis_colour), (heads, axis_colour)]


def k_space(ax: Axes) -> solids.View:
    """The square view of k-space about the origin, the same for every medium, as many pixels across as the axes'
    room holds."""
    return fitted(solids.View(solids.aim(solids.origin, elevation, azimuth), np.full(2, half_width), 1), ax)


def sheet_sides(sheets: core.Vector, view: solids.View) -> list[tuple[meshes.Samples, float]]:
    """Both sheets, translucent, with grid lines along every few latitudes and longitudes."""
    n_latitudes, n_longitudes, n_sheets = sheets.shape
    faces = sheet_faces(n_latitudes, n_longitudes)
    # Each vertex's place on the grid, latitude and longitude, for the grid lines inked on the sheet.
    indices = np.stack(np.meshgrid(np.arange(n_latitudes), np.arange(n_longitudes), indexing="ij"), axis=-1)
    grid = indices.reshape(-1, 2)[faces]                                         # [faces, 3, 2]
    # Each sheet's near side and far side a mesh of its own, so that the far side shows through the near.
    sides = []
    for index, colour in zip(range(n_sheets), map(to_rgb, sheet_colours)):
        places = gpu_points(sheets[:, :, index].reshape(-1))                     # [vertices] Point-
        normals = meshes.smooth(places, faces)
        sides += [(meshes.samples(places, faces, grid, normals, view, face_values=np.zeros(4),
                                  colour=side_colour(np.array(colour), n_latitudes, near),
                                   gloss=0.2), 1.0) for near in (True, False)]
    return sides


def surface(ax: Axes, picture: np.ndarray, view: solids.View) -> AxesImage:
    """A picture of k-space, its axes named at their tips; the image, for an animation to set its data."""
    shown = show(ax, picture, view)
    # Each axis named past its tip, along the direction square to the plane across it.
    label_places = solids.origin + label_distance * solids.normal_direction(solids.axes)   # [axes] Point-
    labels(ax, view, label_places, ["$k_x$", "$k_y$", "$k_z$"], ha="center", va="center", color=axis_colour)
    return shown


def section_axes(ax: Axes) -> list[Line2D]:
    """Axes for the sheets' sections, and each sheet's curve, empty, for its points to be set."""
    curves = [ax.plot([], [], color=colour, linestyle=style)[0]
              for colour, style in zip(sheet_colours, ("-", "--"))]
    ax.axhline(0, color="#cbd5e1", linewidth=charts.FRAME_WIDTH)
    ax.axvline(0, color="#cbd5e1", linewidth=charts.FRAME_WIDTH)
    # The axes named at their ends, without scales: the sheets' shapes against each other are the point.
    ax.text(view_limit, 0, "$k_x$", ha="right", va="bottom", color=axis_colour)
    ax.text(0, view_limit, "$k_z$", ha="left", va="top", color=axis_colour)
    ax.set(xlim=(-view_limit, view_limit), ylim=(-view_limit, view_limit))
    ax.set_aspect("equal")
    ax.set_axis_off()
    return curves


def trace_sections(curves: list[Line2D], sections: core.Vector) -> None:
    """Each sheet's section in the xz plane along its curve."""
    points = coordinates(sections)                                               # [section_samples, sheets, 3]
    for mode, curve in enumerate(curves):
        curve.set_data(points[:, mode, 0], points[:, mode, 2])


@charts.style()
def comparison(sheets: core.Vector, sections: core.Vector) -> plt.Figure:
    """Each medium's sheets above their sections, one medium to a column."""
    count = len(sheets)
    figure, (surface_axes, sections_axes) = charts.subplots(*comparison_size, nrows=2, ncols=count, squeeze=False,
                                                            layout="constrained")
    for i, ax in enumerate(sections_axes):
        surface_axes[i].set_axis_off()
        trace_sections(section_axes(ax), sections[i])
    # The sheets traced once every axes is laid out, the sheets' as bare as their pictures leave them, each at the
    # size of its room.
    for i, ax in enumerate(surface_axes):
        view = k_space(ax)
        surface(ax, solids.draw(k_axes(view), view, np.ones(3), inks=[], translucent=[],
                                meshes=sheet_sides(sheets[i], view)), view)
    return figure


def animate(frames: Iterable[tuple[core.Vector, core.Vector]]) -> Iterator[np.ndarray]:
    """The sheets beside their sections, frame by frame, the frame of k-space and the sections' axes set up once."""
    # The charts' style held while the frames are drawn, for the artists that drawing makes.
    with charts.style():
        figure, (surface_axes, sections_axes) = charts.subplots(*animation_size, ncols=2, layout="constrained")
        # The sheets' axes as bare as their picture leaves them, for the layout to give them their room.
        surface_axes.set_axis_off()
        curves = section_axes(sections_axes)
        view = k_space(surface_axes)
        scene = Scene(view, np.ones(3), k_axes(view), [], inks=[], translucent=[])
        image = surface(surface_axes, np.full((*view.shape, 3), 255, dtype=np.uint8), view)
        # Laid out once, so that nothing moves between frames.
        figure.canvas.draw()
        figure.set_layout_engine("none")
        for sheets, sections in frames:
            image.set_data(scene.draw([], inks=[], translucent=[], meshes=sheet_sides(sheets, view)))
            trace_sections(curves, sections)
            yield capture(figure)
        plt.close(figure)
