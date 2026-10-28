"""The drawings show the spectra of matching films, polarization-selective reflection by a twisted crystal, a
pulse through it, and axion interfaces."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb, to_rgb

from numga import stack
from examples.optics.coating import core
from examples.util.wgsl import solids, strokes
from examples.util.wgsl.scene import Scene

case_colours = ("#2563eb", "#ea580c", "#0d9488", "#7c3aed")
# The quantity plots: one panel and two panels side by side, in inches at 100 dpi, their text size in points
# and their curves' width.
single_size, double_size = (3.6, 2.4), (5.6, 2.4)
font_size, curve_width = 7, 1.2
sm = solids.mv
# The pulse's view: elevation and azimuth in radians, its margin and the pixels across it. The field drawn
# `field_length` nm across the line per unit; the stretch of line framed, as shares of its length cut from its
# start and its end; the widths of the field's path, its spokes, the optic axes and the line, in pixels.
pulse_elevation, pulse_azimuth = np.radians(6.0), np.radians(30.0)
pulse_margin, n_pulse_pixels = 0.02, 300
field_length = 3000.0
framed_start, framed_end = 0.08, 0.12
path_width, spoke_width, optic_width, line_width = 1.25, 0.5, 0.75, 0.75
# The spokes' share of the path's strength, and the least strength a spoke is drawn at; the optic axes' half length, the slab's half width across the line, both in field units, and the slab's colour
# and opacity; the line's colour and the optic axes'.
spoke_opacity, spoke_floor = 0.5, 0.02
optic_half_length, slab_half_width, slab_opacity = 0.6, 1.0, 0.22
slab_colour = np.array(to_rgb("#b8c4d4"))
line_colour, optic_colour = np.array([0.75, 0.76, 0.78]), np.array([0.62, 0.63, 0.65])
# The beam's frame turned into the scene's, whose z is up: a half turn about the line along x + z through the origin
# carries the line, along z, onto x, across the picture, and the field's x onto z, up.
upright = ((sm.yz + sm.xy) * (np.pi / (2 * np.sqrt(2)))).exp()


# --- plumbing -------------------------------------------------------------------------
def quantity_axes(ax: plt.Axes) -> None:
    """Axes of a quantity plot at the plots' text size, without the frame's top and right."""
    ax.tick_params(labelsize=font_size)
    ax.xaxis.label.set_size(font_size)
    ax.yaxis.label.set_size(font_size)
    ax.spines[["top", "right"]].set_visible(False)


def electric_along_x(fields: core.Bivector) -> np.ndarray:
    """The electric field along x of complex fields, read by the plane x sweeps through time, in phase and a
    quarter cycle on `[2, ...]`."""
    mv = fields.real().context.multivector
    along_x = mv.x.dual() & -mv.t.dual()
    return np.stack([(fields.real() & along_x).to_array(), ((-1j * fields).real() & along_x).to_array()])


def spectra(wavelengths: np.ndarray, reflectance: core.Scalar, names: tuple[str, ...]) -> plt.Figure:
    """Reflected power as a percentage of the incident power for each coating."""
    figure, ax = plt.subplots(figsize=single_size, layout="constrained")
    ax.set_prop_cycle(color=case_colours)
    for spectrum, name in zip(reflectance.real().to_array(), names):
        ax.plot(wavelengths, spectrum * 100, linewidth=curve_width, label=name)
    ax.set(xlim=(wavelengths[0], wavelengths[-1]), ylim=(0, None),
           xlabel="wavelength (nm)", ylabel="reflectance (%)")
    quantity_axes(ax)
    ax.legend(fontsize=font_size, frameon=False)
    return figure


def angular(wavelengths: np.ndarray, angles: np.ndarray, reflectance: core.Scalar) -> plt.Figure:
    """The two polarizations over wavelength and incidence angle, on one reflectance scale from none, black,
    to all, white."""
    figure, panels = plt.subplots(1, 2, figsize=double_size, sharex=True, sharey=True,
                                 layout="constrained")
    for ax, spectrum in zip(panels, reflectance.real().to_array()):
        ax.pcolormesh(wavelengths, np.degrees(angles), spectrum,
                      shading="nearest", cmap="magma", vmin=0, vmax=1, rasterized=True)
        ax.set(xlabel="wavelength (nm)")
        quantity_axes(ax)
    panels[0].set_ylabel("incidence angle (degrees)", fontsize=font_size)
    return figure


def field(depths: np.ndarray, fields: core.Bivector) -> plt.Figure:
    """The signed electric field along x of the fields `[cases, layers, samples]` at the depths
    `[layers, samples]`, over two optical cycles."""
    n_cycles = 2
    n_samples_per_cycle = 80
    times = np.linspace(0, n_cycles, n_cycles * n_samples_per_cycle + 1)
    phase = 2 * np.pi * times
    depths = depths.reshape(-1)
    in_phase, quadrature = electric_along_x(fields).reshape(2, fields.shape[0], -1)
    # The real field a phase later: the complex field's real part once its positive imaginary phase has grown.
    waves = (in_phase[:, None, :] * np.cos(phase)[None, :, None]
             - quadrature[:, None, :] * np.sin(phase)[None, :, None])
    limit = np.max(np.abs(waves))
    figure, panels = plt.subplots(1, 2, figsize=double_size, sharex=True, sharey=True,
                                 layout="constrained")
    for ax, wave in zip(panels, waves):
        ax.pcolormesh(depths, times, wave, shading="nearest", cmap="RdBu_r", vmin=-limit, vmax=limit,
                      rasterized=True)
        ax.set(xlabel="depth (nm)")
        quantity_axes(ax)
    # Whole optical cycles only; the colour is the field's sign and relative strength.
    panels[0].set(yticks=np.arange(n_cycles + 1))
    panels[0].set_ylabel("optical cycles", fontsize=font_size)
    return figure


def handedness(wavelengths: np.ndarray, reflectance: core.Scalar, polarizations: tuple[str, ...]) -> plt.Figure:
    """Opposite material twists side by side, `[polarizations, twists, wavelengths]`, with the same
    colours for the incident polarizations."""
    figure, axes = plt.subplots(1, 2, figsize=double_size, sharex=True, sharey=True,
                               layout="constrained")
    values = reflectance.real().to_array()
    for helix, ax in enumerate(axes):
        ax.set_prop_cycle(color=case_colours)
        for spectrum, polarization in zip(values[:, helix], polarizations):
            ax.plot(wavelengths, spectrum * 100, linewidth=curve_width, label=polarization)
        ax.set(xlim=(wavelengths[0], wavelengths[-1]), ylim=(0, 100),
               xlabel="wavelength (nm)")
        quantity_axes(ax)
    axes[0].set_ylabel("reflectance (%)", fontsize=font_size)
    axes[1].legend(fontsize=font_size, frameon=False)
    return figure


def axion(jumps: np.ndarray, reflectance: core.Scalar, names: tuple[str, ...]) -> plt.Figure:
    """Reflected power in each polarization channel across an axion discontinuity."""
    figure, ax = plt.subplots(figsize=single_size, layout="constrained")
    ax.set_prop_cycle(color=case_colours)
    for channel, name in zip(reflectance.real().to_array(), names):
        ax.plot(jumps, channel * 100, linewidth=curve_width, label=name)
    # The jump is marked only where it vanishes, and with it the reflection.
    ax.set(xlim=(jumps[0], jumps[-1]), ylim=(0, None), xticks=[0],
           xlabel="axion jump", ylabel="reflectance (%)")
    quantity_axes(ax)
    ax.legend(fontsize=font_size, frameon=False)
    return figure


def scene(places: np.ndarray) -> solids.Point:
    """Places in the beam's frame, across the line along x and y and along it by depth `[..., 3]`, as points turned
    upright into the scene `[...]`."""
    return upright >> sm.point(places)


def faded(colours: np.ndarray, strength: np.ndarray) -> np.ndarray:
    """Colours `[..., 3]` faded towards the white page as their strength `[...]` falls to zero."""
    return 1 - strength[..., None] * (1 - colours)


def pulse(nodes: core.Direction, slab: core.Direction, directors: core.Direction,
          electric: Iterable[core.Bivector]) -> Iterator[np.ndarray]:
    """The electric field along the line `[nodes]` at each frame, each drawn as it comes, seen from aside and a little
    above: the path of its tip, coloured by its direction around the line, over faint spokes from the line out to it.
    A field that turns traces loops; one that does not stays in its plane. The slab is a translucent box, its optic
    axes grey rods."""
    mv = nodes.context.multivector
    # The plane z reads the depth along the line; x and y read the optic axes across it, and the planes x and y
    # sweep through time read the electric field along them.
    across = stack((mv.x, mv.y))
    swept = stack((mv.x.dual(), mv.y.dual())) & -mv.t.dual()
    depths = (mv.z & nodes).to_array()                                                # [nodes]
    slab_depths = (mv.z & slab).to_array()                                            # [slab nodes]
    optic_axes = optic_half_length * field_length * (across & directors[::3, None]).to_array()   # [rods, 2]
    rod_depths = slab_depths[::3, None]                                              # [rods, 1]

    # The stretch of line framed, the box of the slab's reach across it.
    first, last = depths[0] + framed_start * np.ptp(depths), depths[-1] - framed_end * np.ptp(depths)
    half_width = slab_half_width * field_length
    corners = np.stack(np.meshgrid([-half_width, half_width], [-half_width, half_width], [first, last], indexing="ij"), axis=-1)
    view = solids.view([upright >> sm.point(corners.reshape(-1, 3))], 0.0, pulse_elevation, pulse_azimuth,
                       pulse_margin, n_pulse_pixels)
    line = solids.rods(upright >> sm.point([0, 0, depths[0]]), upright >> sm.point([0, 0, depths[-1]]),
                       line_width / 2 * view.pixel)[None]
    # The slab: a box about its middle on the line, turned upright with everything else.
    middle = sm.point([0, 0, (slab_depths[0] + slab_depths[-1]) / 2])
    box = solids.boxes(upright * solids.translator(middle - solids.origin),
                       np.array([half_width, half_width, np.ptp(slab_depths) / 2]))[None]
    optics = solids.rods(upright >> sm.point(np.concatenate([-optic_axes, rod_depths], axis=-1)),
                         upright >> sm.point(np.concatenate([optic_axes, rod_depths], axis=-1)),
                         optic_width / 2 * view.pixel)

    # The line, the optic axes and the slab stay put.
    picture = Scene(view, np.ones(3), [(line, line_colour), (optics, optic_colour)], [],
                    inks=[], translucent=[(box, slab_colour, slab_opacity)])
    for planes in electric:
        field = (planes[:, None] & swept).to_array()                                 # [nodes, 2]
        tips = scene(np.concatenate([field_length * field, depths[:, None]], axis=-1))   # [nodes] Point-
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
        yield picture.draw([], inks=[], translucent=[], meshes=[(spokes, 1.0), (path, 1.0)])
