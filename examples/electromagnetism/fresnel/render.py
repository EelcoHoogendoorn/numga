"""Ray-traced 3D views and 2D meridian strokes display the two Fresnel wave-surface sheets."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import to_rgb
from matplotlib.image import AxesImage

from numga import stack
from numga.render import Camera, Scene, drawing, meshes, read, simplices, solids
from numga.render.drawing import Draw
from numga.topology import Topology

from examples.electromagnetism.fresnel import core
from examples.util import charts
from examples.util.animation import capture
from examples.util.panels import fitted, labels, offset_panel, overlay, show

sheet_colours = ("#2563eb", "#ea580c")
view_limit = 2.4
latitude_stride = 4
longitude_stride = 8
# Project each spacetime wavevector onto its spatial tip in the rest observer's k-space.
space_tip = core.mv.t ^ (core.Vector.dual() & core.time)
sheet_opacity, grid_opacity = 0.12, 0.4
axis_colour = np.array([0.45, 0.47, 0.52])
elevation, azimuth = np.radians(22.0), np.radians(37.0)
half_width = 2.3
grid_width, arrow_width, tip_radius = 1.0, 1.0, 2.0
axis_length, label_distance = 1.95, 2.12
comparison_size, animation_size = (600, 400), (520, 260)
cross_colour = np.array(to_rgb("#cbd5e1"))
section_dashes = (5.5, 2.4)

grid_shader = """
fn colour(values: vec4<f32>, face_values: vec4<f32>, corner: vec3<f32>, front: bool) -> vec4<f32> {
    let nearest = round(values.xy / strides) * strides;
    let steps = oversampling * fwidth(values.xy);
    var ink = clamp(grid_width / 2 + 0.5 - abs(values.xy - nearest) / steps, vec2<f32>(0.0), vec2<f32>(1.0));
    // Suppress parallel strokes at both poles where latitudes collapse to points.
    ink.x *= select(0.0, 1.0, nearest.x > 0 && nearest.x < f32(n_latitudes - 1));
    let visibility = select(back_shown, front_shown, front);
    return vec4<f32>(sheet_colour, visibility * mix(sheet_opacity, grid_opacity, max(ink.x, ink.y)));
}
"""


def sheet_faces(n_latitudes: int, n_longitudes: int) -> np.ndarray:
    """Interior quads of a `[n_latitudes, n_longitudes]` latitude-longitude grid split into triangle pairs while polar quads form single triangles (`[F, 3]`)."""
    index = np.arange(n_latitudes * n_longitudes).reshape(n_latitudes, n_longitudes)
    corners = np.stack([index[:-1, :-1], index[1:, :-1], index[1:, 1:], index[:-1, 1:]], axis=-1)
    return np.concatenate([corners[:-1, :, [0, 1, 2]].reshape(-1, 3), corners[1:, :, [0, 2, 3]].reshape(-1, 3)])


def side_colour(colour: np.ndarray, n_latitudes: int, near: bool) -> meshes.Colour:
    """The gridded WGSL shader colours either the camera-facing (`near=True`) or far (`near=False`) pass of a translucent sheet."""
    return meshes.Colour(
        grid_shader,
        dict(
            sheet_colour=colour,
            strides=np.array([latitude_stride, longitude_stride]),
            grid_width=grid_width,
            n_latitudes=n_latitudes,
            sheet_opacity=sheet_opacity,
            grid_opacity=grid_opacity,
            front_shown=float(near),
            back_shown=float(not near),
        ),
    )


def wavevector_axes(view: solids.View) -> list[tuple[solids.Quadric, np.ndarray]]:
    """Rods from the origin to ball-tipped endpoints form the three spatial wavevector axes."""
    tips = solids.origin + axis_length * solids.normal_direction(solids.axes)       # [3] Point-
    return [
        (solids.rods(solids.origin, tips, arrow_width / 2 * view.pixel), axis_colour),
        (solids.ball(tips, tip_radius * view.pixel)[:, None], axis_colour),
    ]


def wavevector_view(ax: Axes) -> solids.View:
    """A fixed-width square camera view frames wavevector space around the origin inside `ax`."""
    return fitted(solids.View(solids.aim(solids.origin, elevation, azimuth), np.full(2, half_width), 1), ax)


def sheet_sides(sheets: core.Vector, view: solids.View) -> list[tuple[Draw, float]]:
    """Near and far passes render both translucent sheets (`[n_latitudes, n_longitudes, 2] Vector+`) with latitude and longitude grid lines."""
    n_latitudes, n_longitudes, n_sheets = sheets.shape
    faces = sheet_faces(n_latitudes, n_longitudes)
    topology = Topology(n_latitudes * n_longitudes, faces)
    V, E, F = topology.n_elements
    indices = np.stack(np.meshgrid(np.arange(n_latitudes), np.arange(n_longitudes), indexing="ij"), axis=-1)
    grid = indices.reshape(V, 2)[faces]                                             # [F, 3, 2]
    # Separate near and far passes allow the far side of each translucent sheet to show through the near side.
    sides = []
    for index, colour in zip(range(n_sheets), map(to_rgb, sheet_colours)):
        places = solids.origin + solids.context.include_from_algebra(space_tip(sheets[:, :, index].reshape(V)))  # [V] Point-
        normals = meshes.smooth(places, topology)
        sides += [
            (
                meshes.samples(
                    places,
                    faces,
                    grid,
                    normals,
                    view,
                    face_values=np.zeros(4),
                    colour=side_colour(np.array(colour), n_latitudes, near),
                    gloss=0.2,
                ),
                1.0,
            )
            for near in (True, False)
        ]
    return sides


def surface_panel(ax: Axes, picture: np.ndarray, view: solids.View) -> AxesImage:
    """Axis labels beyond the three arrow tips annotate the rendered wavevector-space image on `ax`."""
    panel_image = show(ax, picture, view)
    label_places = solids.origin + label_distance * solids.normal_direction(solids.axes)  # [3] Point-
    labels(ax, view, label_places, ["kx", "ky", "kz"], ha="center", va="center", color=axis_colour)
    return panel_image


def section_axes(ax: Axes) -> None:
    """Equal-aspect limits and endpoint labels frame the xz meridian section panel without tick scales."""
    ax.text(view_limit, 0, "kx", ha="right", va="bottom", color=axis_colour)
    ax.text(0, view_limit, "kz", ha="left", va="top", color=axis_colour)
    ax.set(xlim=(-view_limit, view_limit), ylim=(-view_limit, view_limit))
    ax.set_aspect("equal")
    ax.set_axis_off()


def section_camera(ax: Axes) -> Camera:
    """An offset-panel camera pairs spacetime wavevectors against the spatial x and z directions."""
    return offset_panel(ax, core.axes[0], core.axes[2])


def section_strokes(sections: core.Vector, camera: Camera) -> np.ndarray:
    """Solid inner and dashed outer meridian strokes (`[n_section_samples, 2] Vector+`) composite over faint origin crosshairs."""
    across_unit, up_unit = (axis.dual() / read(axis, axis.dual()) for axis in (camera.across, camera.up))
    ends = view_limit * stack((-across_unit, across_unit, -up_unit, up_unit))       # [4] Vector+
    loop = Topology.loops([sections.shape[0]])
    return drawing.composited([
        simplices.strokes(ends, Topology.chains([2, 2]), cross_colour, charts.FRAME_PX, camera, cap=simplices.Cap.BUTT),
        simplices.strokes(sections[:, 0], loop, np.array(to_rgb(sheet_colours[0])), charts.LINE_PX, camera),
        simplices.strokes(
            sections[:, 1],
            loop,
            np.array(to_rgb(sheet_colours[1])),
            charts.LINE_PX,
            camera,
            dashes=section_dashes,
            cap=simplices.Cap.BUTT,
        ),
    ])


@charts.style()
def comparison_figure(sheets: core.Vector, sections: core.Vector) -> plt.Figure:
    """One column per medium places the 3D wave surface above its xz meridian section."""
    n_columns = len(sheets)
    figure, (surface_axes, sections_axes) = charts.subplots(
        *comparison_size, nrows=2, ncols=n_columns, squeeze=False, layout="constrained"
    )
    for column, ax in enumerate(sections_axes):
        surface_axes[column].set_axis_off()
        section_axes(ax)
    for column, ax in enumerate(surface_axes):
        view = wavevector_view(ax)
        picture = solids.draw(
            wavevector_axes(view), view, np.ones(3), inks=[], translucent=[], meshes=sheet_sides(sheets[column], view)
        )
        surface_panel(ax, picture, view)
    for column, ax in enumerate(sections_axes):
        overlay(ax, [section_strokes(sections[column], section_camera(ax))])
    return figure


def sheet_animation(frames: Iterable[tuple[core.Vector, core.Vector]]) -> Iterator[np.ndarray]:
    """Yields RGB frames showing the 3D wave surface beside its xz meridian section for each `(sheets, sections)` state."""
    with charts.style():
        figure, (surface_axes, sections_axes) = charts.subplots(*animation_size, ncols=2, layout="constrained")
        surface_axes.set_axis_off()
        section_axes(sections_axes)
        view = wavevector_view(surface_axes)
        scene = Scene(view, np.ones(3), wavevector_axes(view), [], inks=[], translucent=[])
        image = surface_panel(surface_axes, np.full((*view.shape, 3), 255, dtype=np.uint8), view)
        # Freeze the constrained layout after the initial draw so panel positions stay fixed across frames.
        figure.canvas.draw()
        figure.set_layout_engine("none")
        camera = section_camera(sections_axes)
        section_image = overlay(sections_axes, [np.zeros(camera.shape + (4,))])
        for sheets, sections in frames:
            image.set_data(scene.draw([], inks=[], translucent=[], meshes=sheet_sides(sheets, view)))
            section_image.set_data(section_strokes(sections, camera))
            yield capture(figure)
        plt.close(figure)
