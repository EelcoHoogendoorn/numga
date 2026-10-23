"""Matching films, polarization-selective reflection by a twisted crystal, a pulse through it, and axion interfaces."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb, to_rgb

from examples.optics.coating import core
from examples.util.wgsl import solids, strokes
from examples.util.wgsl.scene import Scene

case_colours = ("#2563eb", "#ea580c", "#0d9488", "#7c3aed")
sm = solids.mv
# The pulse's view: elevation and azimuth in radians, its margin and the pixels across it. The field drawn
# `field_length` nm across the line per unit; the stretch of line framed, as shares of its length cut from its
# start and its end; the widths of the field's path, its spokes, the optic axes and the line, in pixels.
pulse_elevation, pulse_azimuth = np.radians(6.0), np.radians(30.0)
pulse_margin, n_pulse_pixels = 0.02, 600
field_length = 3000.0
framed_start, framed_end = 0.08, 0.12
path_width, spoke_width, optic_width, line_width = 2.5, 1.0, 1.5, 1.5
# The spokes' share of the path's strength, and the least strength a spoke is drawn at; the optic axes' half length, the slab's half width across the line, both in field units, and the slab's colour
# and opacity; the line's colour and the optic axes'.
spoke_opacity, spoke_floor = 0.5, 0.02
optic_reach, slab_reach, slab_opacity = 0.6, 1.0, 0.22
slab_colour = np.array(to_rgb("#b8c4d4"))
line_colour, optic_colour = np.array([0.75, 0.76, 0.78]), np.array([0.62, 0.63, 0.65])
# The beam's frame turned into the scene's, whose z is up: a half turn about the line along x + z through the origin
# carries the line, along z, onto x, across the picture, and the field's x onto z, up.
upright = ((sm.yz + sm.xy) * (np.pi / (2 * np.sqrt(2)))).exp()


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


def faded(colours: np.ndarray, strength: np.ndarray) -> np.ndarray:
    """Colours `[..., 3]` faded towards the white page as their strength `[...]` falls to zero."""
    return 1 - strength[..., None] * (1 - colours)


def pulse(nodes: core.Vector, slab: core.Vector, directors: core.Vector,
          electric: Iterable[core.Bivector]) -> Iterator[np.ndarray]:
    """The electric field along the line `[nodes]` at each frame, each drawn as it comes, seen from aside and a little
    above: the path of its tip, coloured by its direction around the line, over faint spokes from the line out to it.
    A field that turns traces loops; one that does not stays in its plane. The slab is a translucent box, its optic
    axes grey rods."""
    algebra = nodes.algebra
    depths = nodes.cast(algebra.subspace("z")).kernel[:, 0]                           # [nodes]
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

    # The line, the optic axes and the slab stay put.
    picture = Scene(view, np.ones(3), [(line, line_colour), (optics, optic_colour)], [],
                    translucent=[(box, slab_colour, slab_opacity)])
    for planes in electric:
        field = planes.cast(algebra.subspace("xt yt")).kernel                        # [nodes, 2]
        tips = scene(np.concatenate([field_length * field, depths[:, None]], axis=-1))   # [nodes, 3]
        hue = (np.arctan2(field[:, 1], field[:, 0]) / (2 * np.pi)) % 1
        strength = np.clip(np.hypot(*field.T) * 4, 0, 1)
        colours = hsv_to_rgb(np.column_stack([hue, np.full_like(hue, 0.85), np.ones_like(hue)]))
        # Every other node's spoke, where the field shows.
        spoked = np.flatnonzero(strength > spoke_floor)[::2]
        spokes = strokes.segments(scene(np.column_stack([np.zeros((len(spoked), 2)), depths[spoked]])), tips[spoked],
                                  faded(colours, spoke_opacity * strength)[spoked], spoke_width, view)
        # Each piece of the path in the colour and strength halfway along it.
        shades = faded(colours, strength)
        path = strokes.segments(tips[:-1], tips[1:], (shades[:-1] + shades[1:]) / 2, path_width, view)
        yield picture.draw(meshes=[(spokes, 1.0), (path, 1.0)])
