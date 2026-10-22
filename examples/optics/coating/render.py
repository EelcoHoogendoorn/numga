"""Rotating optic axes, polarization-selective reflection, and axion interfaces."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb, to_rgb

from examples.optics.coating import core
from examples.wgsl import labels, meshes, show, solids

case_colours = ("#2563eb", "#ea580c", "#0d9488", "#7c3aed")
sm = solids.mv
# The twist's view: elevation and azimuth in radians, azimuth zero looking along +y; its margin, a share of the
# larger side, and the pixels across it.
twist_elevation, twist_azimuth = np.radians(16.0), np.radians(35.0)
twist_margin, n_twist_pixels = 0.06, 360
# The widths of the optic axes, the helix's axis and the depth scale, the scale's arrowhead and its ticks' radius, in
# pixels; the scale's tick spacing in nm, and how far the scale, its numbers and its name lie, in rod half lengths.
rod_width, axis_width, scale_width, head_length, head_width, tick_radius = 3.0, 3.0, 3.0, 12.0, 7.0, 3.5
depth_tick = 500.0
scale_distance, number_distance, title_height = 1.3, 0.2, 0.25
scale_colour, label_colour = np.array([0.6, 0.62, 0.66]), np.array([0.3, 0.32, 0.36])
# The pulse's view: elevation and azimuth in radians, its margin and the pixels across it. The field drawn
# `field_length` nm across the line per unit; the stretch of line framed, as shares of its length cut from its
# start and its end; the widths of the field's path, its spokes, the optic axes and the line, in pixels.
pulse_elevation, pulse_azimuth = np.radians(6.0), np.radians(30.0)
pulse_margin, n_pulse_pixels = 0.02, 600
field_length = 3000.0
framed_start, framed_end = 0.08, 0.12
path_width, spoke_width, optic_width, line_width = 2.5, 1.0, 1.5, 1.5
# The tube's sides around the path; the spokes' share of the path's opacity, and the least strength a spoke is drawn
# at; the optic axes' half length, the slab's half width across the line, both in field units, and the slab's colour
# and opacity; the line's colour and the optic axes'.
n_around = 6
spoke_opacity, spoke_floor = 0.5, 0.02
optic_reach, slab_reach, slab_opacity = 0.6, 1.0, 0.22
slab_colour = np.array(to_rgb("#b8c4d4"))
line_colour, optic_colour = np.array([0.75, 0.76, 0.78]), np.array([0.62, 0.63, 0.65])
# The beam's frame turned into the scene's, whose z is up: a half turn about the line along x + z through the origin
# carries the line, along z, onto x, across the picture, and the field's x onto z, up.
upright = ((sm.yz + sm.xy) * (np.pi / (2 * np.sqrt(2)))).exp()
# Each vertex's own colour and how much of what lies behind it it hides, blended across each face.
translucent = """
fn colour(values: vec4<f32>, face_values: vec4<f32>, corner: vec3<f32>, front: bool) -> vec4<f32> {
    return values;
}
"""


# --- plumbing -------------------------------------------------------------------------
def spectra(wavelengths: np.ndarray, reflectance: core.Scalar, names: tuple[str, ...]) -> plt.Figure:
    """Reflected power as a percentage of the incident power for each coating."""
    figure, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    ax.set_prop_cycle(color=case_colours)
    values = reflectance.real().cast(reflectance.algebra.subspace.scalar()).kernel[..., 0]
    for spectrum, name in zip(values, names):
        ax.plot(wavelengths, spectrum * 100, linewidth=1.8, label=name)
    ax.set(xlim=(wavelengths[0], wavelengths[-1]), ylim=(0, None),
           xlabel="wavelength (nm)", ylabel="reflectance (%)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9, frameon=False)
    return figure


def angular(wavelengths: np.ndarray, angles: np.ndarray, reflectance: core.Scalar) -> plt.Figure:
    """The two polarizations over wavelength and incidence angle, on one reflectance scale."""
    figure, panels = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True, sharey=True,
                                 layout="constrained")
    values = reflectance.real().cast(reflectance.algebra.subspace.scalar()).kernel[..., 0]
    for ax, spectrum in zip(panels, values):
        image = ax.pcolormesh(wavelengths, np.degrees(angles), spectrum,
                              shading="nearest", cmap="magma", vmin=0, vmax=1, rasterized=True)
        ax.set(xlabel="wavelength (nm)")
    panels[0].set_ylabel("incidence angle (degrees)")
    figure.colorbar(image, ax=panels, label="reflectance")
    return figure


def field(depths: np.ndarray, fields: core.Bivector) -> plt.Figure:
    """The signed electric x component of the fields, its plane `xt`, `[cases, layers, samples]` at the
    depths `[layers, samples]`, over two optical cycles."""
    cycles = 2
    samples_per_cycle = 80
    times = np.linspace(0, cycles, cycles * samples_per_cycle + 1)
    phase = 2 * np.pi * times
    depths = depths.reshape(-1)
    in_phase = fields.real().cast(fields.algebra.subspace("xt")).kernel[..., 0].reshape(fields.shape[0], -1)
    quadrature = (-1j * fields).real().cast(fields.algebra.subspace("xt")).kernel[..., 0].reshape(fields.shape[0], -1)
    waves = (in_phase[:, None, :] * np.cos(phase)[None, :, None]
             - quadrature[:, None, :] * np.sin(phase)[None, :, None])
    limit = np.max(np.abs(waves))
    figure, panels = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True, sharey=True,
                                 layout="constrained")
    for ax, wave in zip(panels, waves):
        ax.pcolormesh(depths, times, wave, shading="nearest", cmap="RdBu_r", vmin=-limit, vmax=limit,
                      rasterized=True)
        ax.set(xlabel="depth (nm)")
    # Whole optical cycles only; the colour is the field's sign and relative strength.
    panels[0].set(ylabel="optical cycles", yticks=np.arange(cycles + 1))
    return figure


def twist(centres: core.Vector, directors: core.Vector) -> plt.Figure:
    """Undirected optic axes at the layer depths, about the helix's axis, beside a depth scale; rod lengths are
    schematic."""
    positions = centres.real().cast(centres.algebra.subspace("x y z")).kernel
    directions = directors.real().cast(directors.algebra.subspace("x y z")).kernel
    half_width = np.ptp(positions[:, 2]) * 0.14
    # The depth scale stands beside the helix, past the rods' reach to the camera's right, ticked every few hundred nm.
    camera = solids.aim(solids.origin, twist_elevation, twist_azimuth)
    right = (solids.axes & (camera >> sm.direction([1, 0, 0]))).to_array()
    scale = scale_distance * half_width * right + np.eye(3)[2] * np.array([[0.0], [positions[:, 2].max()]])   # [2, 3]
    ticks = scale[0] + np.eye(3)[2] * np.arange(depth_tick, scale[1, 2], depth_tick)[:, None]          # [ticks, 3]
    numbers = ticks + number_distance * half_width * right
    title = scale[1:] + title_height * half_width * np.eye(3)[2]                                       # [1, 3]
    starts, ends = sm.point(positions - half_width * directions), sm.point(positions + half_width * directions)
    view = solids.view([starts, ends, sm.point(numbers), sm.point(title)], 0.0, twist_elevation, twist_azimuth,
                       twist_margin, n_twist_pixels)

    shafts, heads = solids.arrows(sm.point(scale[0]), sm.point(scale[1]), scale_width / 2 * view.pixel,
                                  head_length * view.pixel, head_width / 2 * view.pixel)
    parts = [(solids.rods(starts, ends, rod_width / 2 * view.pixel), np.array(to_rgb(case_colours[0]))),
             (solids.rods(sm.point(positions[0]), sm.point(positions[-1]), axis_width / 2 * view.pixel)[None],
              np.array(to_rgb("#cbd5e1"))),
             (shafts[None], scale_colour), (heads[None], scale_colour),
             (solids.ball(sm.point(ticks), tick_radius * view.pixel)[:, None], scale_colour)]
    figure = plt.figure(figsize=(view.shape[1] / 100, view.shape[0] / 100), dpi=100)
    ax = figure.add_axes((0, 0, 1, 1))
    show(ax, solids.draw(parts, view, np.ones(3)), view)
    labels(ax, view, sm.point(numbers), [f"{tick[2]:g}" for tick in ticks], ha="left", va="center", fontsize=9,
           color=label_colour)
    labels(ax, view, sm.point(title), ["depth (nm)"], ha="center", va="bottom", fontsize=10, color=label_colour)
    return figure


def handedness(wavelengths: np.ndarray, reflectance: core.Scalar, polarizations: tuple[str, ...]) -> plt.Figure:
    """Opposite material twists side by side, `[polarizations, twists, wavelengths]`, with the same
    colours for the incident polarizations."""
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True, sharey=True,
                               layout="constrained")
    values = reflectance.real().cast(reflectance.algebra.subspace.scalar()).kernel[..., 0]
    for helix, ax in enumerate(axes):
        ax.set_prop_cycle(color=case_colours)
        for spectrum, polarization in zip(values[:, helix], polarizations):
            ax.plot(wavelengths, spectrum * 100, linewidth=1.8, label=polarization)
        ax.set(xlim=(wavelengths[0], wavelengths[-1]), ylim=(0, 100),
               xlabel="wavelength (nm)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("reflectance (%)")
    axes[1].legend(fontsize=9, frameon=False)
    return figure


def axion(jumps: np.ndarray, reflectance: core.Scalar, names: tuple[str, ...]) -> plt.Figure:
    """Reflected power in each polarization channel across an axion discontinuity."""
    figure, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    ax.set_prop_cycle(color=case_colours)
    values = reflectance.real().cast(reflectance.algebra.subspace.scalar()).kernel[..., 0]
    for channel, name in zip(values, names):
        ax.plot(jumps, channel * 100, linewidth=1.8, label=name)
    # The jump is marked only where it vanishes, and with it the reflection.
    ax.set(xlim=(jumps[0], jumps[-1]), ylim=(0, None), xticks=[0],
           xlabel="axion jump", ylabel="reflectance (%)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9, frameon=False)
    return figure


def scene(places: np.ndarray) -> np.ndarray:
    """Places in the beam's frame, across the line along x and y and along it by depth `[..., 3]`, turned upright
    into the scene `[..., 3]`."""
    return (solids.axes & (upright >> sm.point(places))[..., None]).to_array()


def strokes(starts: np.ndarray, ends: np.ndarray, colours: np.ndarray, width: float,
            view: solids.View) -> tuple[np.ndarray, np.ndarray]:
    """Segments between places in the scene `[segments, 3]` as tubes `width` pixels across, each its colour and how
    much of what lies behind it it hides `[segments, 4]`, at every sample, as `meshes.samples`."""
    rings, faces = meshes.tube(np.stack([starts, ends], axis=1), width / 2 * view.pixel, n_around)
    # Each segment's faces in a run of their own, in the order of the segments.
    values = np.repeat(colours, len(faces) // len(starts), axis=0)[:, None]
    return meshes.samples(rings, faces, values, meshes.smooth(rings, faces), view, colour=translucent)


def tube(rings: solids.Point, faces: np.ndarray, colours: np.ndarray, view: solids.View) -> tuple[np.ndarray, np.ndarray]:
    """A polyline's tube, its rings of vertices and its triangles as `meshes.tube` builds them, its colour and how much
    it hides given at each vertex of the polyline `[n, 4]` and blended between, at every sample, as `meshes.samples`."""
    # The tube's vertices run ring after ring, one ring about each vertex of the polyline.
    return meshes.samples(rings, faces, colours[faces // n_around], meshes.smooth(rings, faces), view,
                          colour=translucent)


def pulse(nodes: core.Vector, slab: core.Vector, directors: core.Vector, electric: core.Bivector) -> list[np.ndarray]:
    """The electric field along the line at each frame, seen from aside and a little above: the path of its tip,
    coloured by its direction around the line, over faint spokes from the line out to it. A field that turns traces
    loops; one that does not stays in its plane. The slab is a translucent box, its optic axes grey rods."""
    algebra = nodes.algebra
    depths = nodes.cast(algebra.subspace("z")).kernel[:, 0]                           # [nodes]
    fields = electric.cast(algebra.subspace("xt yt")).kernel                         # [frames, nodes, 2]
    slab_depths = slab.cast(algebra.subspace("z")).kernel[:, 0]                       # [slab nodes]
    optic_axes = optic_reach * field_length * directors.cast(algebra.subspace("x y")).kernel[::3]   # [rods, 2]
    rod_depths = slab_depths[::3, None]                                              # [rods, 1]

    # The stretch of line framed, the box of the slab's reach across it.
    first, last = depths[0] + framed_start * np.ptp(depths), depths[-1] - framed_end * np.ptp(depths)
    reach = slab_reach * field_length
    corners = np.stack(np.meshgrid([-reach, reach], [-reach, reach], [first, last], indexing="ij"), axis=-1)
    view = solids.view([upright >> sm.point(corners.reshape(-1, 3))], 0.0, pulse_elevation, pulse_azimuth,
                       pulse_margin, n_pulse_pixels)
    line = solids.rods(upright >> sm.point([0, 0, depths[0]]), upright >> sm.point([0, 0, depths[-1]]),
                       line_width / 2 * view.pixel)[None]
    # The slab: a box about its middle on the line, turned upright with everything else.
    middle = sm.point([0, 0, (slab_depths[0] + slab_depths[-1]) / 2])
    box = solids.boxes(upright * solids.translator(middle - solids.origin),
                       np.array([reach, reach, np.ptp(slab_depths) / 2]))[None]
    optics = solids.rods(upright >> sm.point(np.concatenate([-optic_axes, rod_depths], axis=-1)),
                         upright >> sm.point(np.concatenate([optic_axes, rod_depths], axis=-1)),
                         optic_width / 2 * view.pixel)

    # Every frame's path at once, so that the tubes' frames walk along the nodes once for all of them; each frame's
    # tube its own run of rings, all with the same triangles.
    paths = scene(np.concatenate([field_length * fields, np.broadcast_to(depths[:, None], fields.shape[:2] + (1,))],
                                 axis=-1))                                          # [frames, nodes, 3]
    rings, faces = meshes.tube(paths, path_width / 2 * view.pixel, n_around)
    per_frame = len(depths) * n_around
    faces = faces[:len(faces) // len(fields)]
    frames = []
    for index, (field, tips) in enumerate(zip(fields, paths)):
        hue = (np.arctan2(field[:, 1], field[:, 0]) / (2 * np.pi)) % 1
        strength = np.clip(np.hypot(*field.T) * 4, 0, 1)
        colours = hsv_to_rgb(np.column_stack([hue, np.full_like(hue, 0.85), np.ones_like(hue)]))
        # Every other node's spoke, where the field shows: a spoke of no length has no direction to build a tube about.
        spoked = np.flatnonzero(strength > spoke_floor)[::2]
        spokes = strokes(scene(np.column_stack([np.zeros((len(spoked), 2)), depths[spoked]])), tips[spoked],
                         np.column_stack([colours, spoke_opacity * strength])[spoked], spoke_width, view)
        path = tube(rings[index * per_frame:(index + 1) * per_frame], faces, np.column_stack([colours, strength]), view)
        frames.append(solids.draw([(line, line_colour), (optics, optic_colour)], view, np.ones(3),
                                  translucent=[(box, slab_colour, slab_opacity)], meshes=[(*spokes, 1.0), (*path, 1.0)]))
    return frames
