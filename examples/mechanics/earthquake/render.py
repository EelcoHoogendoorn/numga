"""Focal mechanism beachballs in the equal-area projection of the lower hemisphere pair with GPU-traced fault patches, slip arrows, and P-wave radiation lobes in space."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb, to_rgba
from matplotlib.image import AxesImage

from numga import concatenate, stack, where
from examples.util.animation import capture
from examples.mechanics.earthquake import core
from numga.render.drawing import compose, composited
from numga.topology import Topology
from numga.render import Camera, contours, meshes, simplices, solids
from examples.util.panels import fitted, offset_panel, overlay, show

# Pushes are shaded and pulls are pale; nodal lines, pressure and tension markers, and slip arrows use distinct hues.
push_colour, pull_colour = "#1e293b", "#f8fafc"
nodal_colour, pressure_colour, tension_colour, slip_colour = "#0f172a", "#0ea5e9", "#e11d48", "#f59e0b"
# Pixel stroke widths and marker radii on the beachball.
nodal_width, ball_rim_width = 1.1, 1.2
axis_radius, axis_rim = 4.0, 0.9
# Surface colours for three-dimensional P-wave lobes and the translucent fault patch.
push_lobe_colour, pull_lobe_colour = "#475569", "#e2e8f0"
fault_colour, rim_colour = "#fde68a", "#b45309"
# Fault disc radius, face opacity, pixel thickness, and slip arrow offsets and length.
patch_radius = 0.55
patch_opacity, patch_thickness = 0.16, 1.0
arrow_offset, arrow_behind, slip_length = 0.15, 0.45, 0.9
# Spherical camera angles (azimuth zero looking along +y) and fractional framing margin.
elevation, azimuth = np.radians([24.0, -60.0 + 90.0])
margin = 0.12
# Pixel widths for the fault rim and slip arrow tubes, plus rim sample count.
rim_width, arrow_width, chevron_length = 2.2, 4.0, 16.0
n_rim = 199
# Half-turn about the northeast bisector maps seismological (north, east, down) into screen (east, north, up).
drawing_turn = (core.down.commutator(core.north - core.east).normalized() * (np.pi / 2)).exp()   # [] Rotor+


def lambert_projection(directions: core.Point) -> core.Point:
    """Projecting unit `directions` (`[...] Point-`) onto the level ground by the Lambert equal-area formula maps upper-hemisphere directions antipodally and shrinks horizontal reach by `sqrt(1 + depth)`."""
    # Level ground faces up, so pairing with `ground` is positive on the upper hemisphere.
    lower = where((core.ground & directions) > 0, -directions, directions)    # [...] Point-
    depth = -(core.ground & lower)                                            # [...] Scalar+
    return (lower - core.down * depth) / (1 + depth).square_root()            # [...] Point-


def ball_grid(directions: core.Point) -> tuple[core.Point, Topology]:
    """Projecting a `[n_polar, n_azimuths]` hemisphere grid onto the drawing plane forms a flat vertex array and quad topology."""
    places = drawing_turn >> lambert_projection(directions).reshape(-1)       # [...] Point-
    return places, Topology.grid(*directions.shape)


def shade_ball(places: core.Point, grid: Topology, push: core.Scalar, camera_view: Camera) -> np.ndarray:
    """Shading positive `push` regions and zero-crossing nodal contours over `grid` produces an RGBA beachball image."""
    values = push.to_array().reshape(-1)
    bound = 2 * np.abs(values).max()
    sides = contours.Bands(-bound, bound, np.array([to_rgba(pull_colour), to_rgba(push_colour)]))
    nodal = contours.Level(0.0, np.array(to_rgba(nodal_colour)), nodal_width)
    return contours.picture(places, grid, values, np.array(to_rgb(pull_colour)), camera_view, levels=(nodal,), bands=(sides,))


def frame_ball(ax) -> None:
    """Configures `ax` as an equal-aspect axisless square framing the unit beachball."""
    ax.set_xlim(-1.12, 1.12)
    ax.set_ylim(-1.12, 1.12)
    ax.set_aspect("equal")
    ax.set_axis_off()


def ball_pictures(places: core.Point, grid: Topology, push: core.Scalar, axes: core.Point,
                  camera_view: Camera) -> list[np.ndarray]:
    """Layers the shaded beachball, outer horizon rim, and pressure and tension axis markers for `camera_view`."""
    # Horizontal directions lie on the equator of the projection and form its circular rim.
    level = (core.horizontal * (-np.linspace(0, 2 * np.pi, n_rim, endpoint=False) / 2)).exp() >> core.north   # [n_rim] Point-
    principal = drawing_turn >> lambert_projection(axes)[[0, 2]]              # [2] Point-
    return [shade_ball(places, grid, push, camera_view),
            simplices.strokes(drawing_turn >> level, Topology.loops([n_rim]), np.array(to_rgb(nodal_colour)),
                              ball_rim_width, camera_view),
            simplices.markers(principal, np.ones(3), axis_radius + axis_rim, camera_view),
            simplices.markers(principal, np.array([to_rgb(pressure_colour), to_rgb(tension_colour)]), axis_radius, camera_view)]


def pin_ball(ax, places: core.Point, grid: Topology, push: core.Scalar,
             axes: core.Point) -> tuple[AxesImage, Camera]:
    """Pins a composited beachball into `ax` and returns its image artist and camera for animation updates."""
    camera_view = offset_panel(ax, core.mv.x, core.mv.y)
    return overlay(ax, ball_pictures(places, grid, push, axes, camera_view)), camera_view


def draw_mechanisms(directions: core.Point, push: core.Scalar, axes: core.Point) -> plt.Figure:
    """Draws the focal mechanism beachballs for a batch of faults side by side."""
    n_faults = push.shape[0]
    figure, panels = plt.subplots(1, n_faults, figsize=(3.2 * n_faults, 3.2), dpi=55, squeeze=False)
    panels = panels[0]
    for ax in panels:
        frame_ball(ax)
    figure.tight_layout()
    places, grid = ball_grid(directions)
    for index, ax in enumerate(panels):
        pin_ball(ax, places, grid, push[index], axes[index])
    return figure


def lobe_surface(directions: core.Point, push: core.Scalar) -> core.Point:
    """Scaling unit `directions` by `push.abs()` and rotating into screen space places the P-wave radiation surface (`[...] Point-`)."""
    return drawing_turn >> (directions * push.abs())                          # [...] Point-


def slip_arrows(normal: core.Point, slip: core.Point) -> tuple[core.Point, core.Point]:
    """Offsetting above and below the fault plane along `normal` places opposing arrow shafts along `slip` (`[..., 2] Point-`)."""
    sides = np.array([1.0, -1.0])                                             # [2]
    normal, slip = drawing_turn >> normal, drawing_turn >> slip               # [...] Point-, [...] Point-
    starts = (normal * arrow_offset - slip * arrow_behind) * sides            # [..., 2] Point-
    return starts, starts + slip * (slip_length * sides)


def scene_view(surfaces: core.Point, ends: core.Point) -> solids.View:
    """Frames P-wave lobe `surfaces` and slip arrow `ends` in a shared three-dimensional orthographic view."""
    framed = [solids.origin + solids.context.include_from_algebra(tips) for tips in (surfaces, ends)]
    return solids.view(framed, 0.0, elevation, azimuth, margin, 1)


def draw_fault(directions: core.Point, push: core.Scalar, normal: core.Point, slip: core.Point,
               fault_view: solids.View) -> np.ndarray:
    """Renders the translucent fault disc, opposing slip arrows, and lit P-wave lobes (`[n_polar, n_azimuths]`) from `fault_view`."""
    n_polar, n_azimuths = push.shape
    surface = lobe_surface(directions, push).reshape(-1)                      # [n_polar * n_azimuths] Point-
    # Split each grid quad into two triangles and colour it by the mean P-wave sign at its four corners.
    index = np.arange(n_polar * n_azimuths).reshape(n_polar, n_azimuths)      # [n_polar, n_azimuths]
    quads = np.stack([index[:-1, :-1], index[1:, :-1], index[1:, 1:], index[:-1, 1:]], axis=-1)   # [n_polar - 1, n_azimuths - 1, 4]
    values = push.to_array()
    middle = values[1:, 1:] + values[:-1, :-1] + values[1:, :-1] + values[:-1, 1:]
    colours = np.where(middle[..., None] > 0, to_rgb(push_lobe_colour), to_rgb(pull_lobe_colour))   # [n_polar - 1, n_azimuths - 1, 3]
    triangles = quads[..., [[0, 1, 2], [0, 2, 3]]].reshape(-1, 3)             # [...]
    colours = np.repeat(colours.reshape(-1, 1, 3), 2, axis=0)                 # [..., 1, 3]
    places = solids.origin + solids.context.include_from_algebra(surface)     # [n_polar * n_azimuths] Point-
    # Omit degenerate triangles collapsed at the poles or pinched into the origin along nodal lines.
    corners = places[triangles]                                               # [..., 3] Point-
    spanned = (corners[:, 0] & corners[:, 1] & corners[:, 2]).norm() > 0
    lobe_samples = meshes.samples(
        places, triangles[spanned], colours[spanned],
        meshes.smooth(places, Topology(len(places), triangles[spanned])),
        fault_view, face_values=np.zeros(4), colour=meshes.plain, gloss=solids.gloss,
    )
    # Build a thin translucent cylinder for the fault patch with an inked circular rim.
    across = solids.context.include_from_algebra(drawing_turn >> normal)      # [] Point-
    faces = solids.origin + stack([-across, across]) * (patch_thickness / 2 * fault_view.pixel)   # [2] Point-
    disc = solids.rods(faces[0], faces[1], patch_radius + rim_width / 2 * fault_view.pixel)[None]   # [1, 3] Plane+ <- Point-
    rim = solids.Ink(0, solids.rods(faces[0], faces[1], patch_radius)[:1], solids.everywhere[None],
                     np.array(to_rgb(rim_colour)), rim_width, 1.0)
    fault_layer = solids.layer([], fault_view, [rim], translucent=[(disc, np.array(to_rgb(fault_colour)), patch_opacity)],
                               meshes=[(lobe_samples, 1.0)])
    starts, ends = (solids.origin + solids.context.include_from_algebra(points) for points in slip_arrows(normal, slip))
    n_sides = starts.shape[0]
    arrows = simplices.tubes(stack((starts, ends), axis=-1).reshape(-1), Topology.chains([2] * n_sides),
                             np.array(to_rgb(slip_colour)), arrow_width, fault_view, gloss=solids.gloss,
                             chevrons=np.ones(n_sides), chevron=chevron_length)
    arrow_colour, arrow_depth = solids.layer([], fault_view, inks=[], translucent=[], meshes=[(arrows, 1.0)])
    # Composite the slip arrows in front so they stay legible through the surrounding radiation lobes.
    return compose([(arrow_colour, np.full_like(arrow_depth, -np.inf), 1.0), (*fault_layer, 1.0)], np.ones(3))


def animate_sweep(hemisphere: core.Point, frames: Iterable[tuple]) -> Iterator[np.ndarray]:
    """Renders each rake-sweep state (`normal, slip, axes, ball, fault_sphere, lobes, ...`) in a side-by-side 3D lobe and 2D beachball view."""
    frames = [state[:6] for state in frames]
    normals, slips, _, _, spheres, pushes = (stack(list(parts)) for parts in zip(*frames))
    figure = plt.figure(figsize=(8.4, 4.2), dpi=50)
    fault_ax, ax = figure.add_subplot(1, 2, 1), figure.add_subplot(1, 2, 2)
    frame_ball(ax)
    fault_ax.set_axis_off()
    figure.tight_layout()
    seen = fitted(scene_view(lobe_surface(spheres, pushes), concatenate(slip_arrows(normals[:, None], slips[:, None]))), fault_ax)
    image = show(fault_ax, np.ones((*seen.shape, 3)), seen)
    places, grid = ball_grid(hemisphere)
    shading, ball_view = pin_ball(ax, places, grid, frames[0][3], frames[0][2])
    for normal, slip, axes, ball, sphere, push in frames:
        image.set_data(draw_fault(sphere, push, normal, slip, seen))
        shading.set_data(composited(ball_pictures(places, grid, ball, axes, ball_view)))
        yield capture(figure)
    plt.close(figure)
