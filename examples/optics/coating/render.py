"""The drawings show the spectra of matching films, polarization-selective reflection by a twisted crystal, a
pulse through it, and axion interfaces."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb, to_rgb

from numga import stack
from examples.optics.coating import core
from examples.util import charts
from examples.util.wgsl import solids, strokes
from examples.util.wgsl.scene import Scene

case_colours = ("#2563eb", "#ea580c", "#0d9488", "#7c3aed")
# The quantity plots' sizes in pixels: one panel, about half the column, and two panels side by side.
single_size, double_size = (420, 280), (640, 280)
sm = solids.mv
# The pulse's view: elevation and azimuth in radians, its margin and the pixels across it. The field drawn
# `field_length` nm across the line per unit; the stretch of line framed, as shares of its length cut from its
# start and its end; the widths of the field's path, its spokes, the optic axes and the line, in pixels.
pulse_elevation, pulse_azimuth = np.radians(6.0), np.radians(30.0)
pulse_margin, n_pulse_pixels = 0.02, 600
field_length = 3000.0
framed_start, framed_end = 0.08, 0.12
path_width, spoke_width, optic_width, line_width = 2.25, 1.1, 1.5, 1.5
# The spokes' share of the path's strength, and the least strength a spoke is drawn at; the optic axes' half length, the slab's half width across the line, both in field units, and the slab's colour
# and opacity; the line's colour and the optic axes'.
spoke_opacity, spoke_floor = 0.5, 0.02
optic_half_length, slab_half_width, slab_opacity = 0.6, 1.0, 0.22
slab_colour = np.array(to_rgb("#b8c4d4"))
line_colour, optic_colour = np.array([0.75, 0.76, 0.78]), np.array([0.62, 0.63, 0.65])
# The beam's frame turned into the scene's, whose z is up: a half turn about the line along x + z through the origin
# carries the line, along z, onto x, across the picture, and the field's x onto z, up.
upright = ((sm.yz + sm.xy) * (np.pi / (2 * np.sqrt(2)))).exp()
# The four corners of a square across the line, as steps along its two directions.
square_corners = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])            # [corners, 2]


# --- plumbing -------------------------------------------------------------------------
def quantity_axes(ax: plt.Axes) -> None:
    """Axes of a quantity plot, without the frame's top and right."""
    ax.spines[["top", "right"]].set_visible(False)


def gpu_points(points: core.Direction) -> solids.Point:
    """The GPU library's own points at the places of the example's, in the beam's frame turned upright into the scene:
    their pairings with the planes x, y and z, the observer's slice of spacetime, are the one hand-off from this
    algebra to the renderer's."""
    mv = points.context.multivector
    return upright >> sm.point((stack((mv.x, mv.y, mv.z)) & points[..., None]).to_array())


@charts.style()
def spectra(wavelengths: np.ndarray, reflectance: core.Scalar, names: tuple[str, ...]) -> plt.Figure:
    """Reflected power as a percentage of the incident power for each coating."""
    figure, ax = charts.subplots(*single_size, layout="constrained")
    ax.set_prop_cycle(color=case_colours)
    for spectrum, name in zip(reflectance.real().to_array(), names):
        ax.plot(wavelengths, spectrum * 100, label=name)
    ax.set(xlim=(wavelengths[0], wavelengths[-1]), ylim=(0, None),
           xlabel="wavelength (nm)", ylabel="reflectance (%)")
    quantity_axes(ax)
    ax.legend(frameon=False)
    return figure


@charts.style()
def angular(wavelengths: np.ndarray, angles: np.ndarray, reflectance: core.Scalar) -> plt.Figure:
    """The two polarizations over wavelength and incidence angle, on one reflectance scale from none, black,
    to all, white."""
    figure, panels = charts.subplots(*double_size, ncols=2, sharex=True, sharey=True,
                                 layout="constrained")
    for ax, spectrum in zip(panels, reflectance.real().to_array()):
        ax.pcolormesh(wavelengths, np.degrees(angles), spectrum,
                      shading="nearest", cmap="magma", vmin=0, vmax=1, rasterized=True)
        ax.set(xlabel="wavelength (nm)")
        quantity_axes(ax)
    panels[0].set_ylabel("incidence angle (degrees)")
    return figure


@charts.style()
def field(depths: np.ndarray, fields: core.Bivector) -> plt.Figure:
    """The signed electric field along x of the fields `[cases, layers, samples]` at the depths
    `[layers, samples]`, over two optical cycles."""
    n_cycles = 2
    n_samples_per_cycle = 80
    times = np.linspace(0, n_cycles, n_cycles * n_samples_per_cycle + 1)
    phase = 2 * np.pi * times
    depths = depths.reshape(-1)
    mv = fields.real().context.multivector
    # The plane x sweeps through time reads the electric field along x.
    along_x = mv.x.dual() & -mv.t.dual()                                     # [] Bivector-
    # The real field a phase later: the complex field's real part once its positive imaginary phase has grown.
    later = (fields.reshape(fields.shape[0], -1)[:, None] * np.exp(1j * phase)[:, None]).real()   # [cases, times, samples] Bivector+
    waves = (later & along_x).to_array()                                     # [cases, times, samples]
    limit = np.max(np.abs(waves))
    figure, panels = charts.subplots(*double_size, ncols=2, sharex=True, sharey=True,
                                 layout="constrained")
    for ax, wave in zip(panels, waves):
        ax.pcolormesh(depths, times, wave, shading="nearest", cmap="RdBu_r", vmin=-limit, vmax=limit,
                      rasterized=True)
        ax.set(xlabel="depth (nm)")
        quantity_axes(ax)
    # Whole optical cycles only; the colour is the field's sign and relative strength.
    panels[0].set(yticks=np.arange(n_cycles + 1))
    panels[0].set_ylabel("optical cycles")
    return figure


@charts.style()
def handedness(wavelengths: np.ndarray, reflectance: core.Scalar, polarizations: tuple[str, ...]) -> plt.Figure:
    """Opposite material twists side by side, `[polarizations, twists, wavelengths]`, with the same
    colours for the incident polarizations."""
    figure, axes = charts.subplots(*double_size, ncols=2, sharex=True, sharey=True,
                               layout="constrained")
    values = reflectance.real().to_array()
    for helix, ax in enumerate(axes):
        ax.set_prop_cycle(color=case_colours)
        for spectrum, polarization in zip(values[:, helix], polarizations):
            ax.plot(wavelengths, spectrum * 100, label=polarization)
        ax.set(xlim=(wavelengths[0], wavelengths[-1]), ylim=(0, 100),
               xlabel="wavelength (nm)")
        quantity_axes(ax)
    axes[0].set_ylabel("reflectance (%)")
    axes[1].legend(frameon=False)
    return figure


@charts.style()
def axion(jumps: np.ndarray, reflectance: core.Scalar, names: tuple[str, ...]) -> plt.Figure:
    """Reflected power in each polarization channel across an axion discontinuity."""
    figure, ax = charts.subplots(*single_size, layout="constrained")
    ax.set_prop_cycle(color=case_colours)
    for channel, name in zip(reflectance.real().to_array(), names):
        ax.plot(jumps, channel * 100, label=name)
    # The jump is marked only where it vanishes, and with it the reflection.
    ax.set(xlim=(jumps[0], jumps[-1]), ylim=(0, None), xticks=[0],
           xlabel="axion jump", ylabel="reflectance (%)")
    quantity_axes(ax)
    ax.legend(frameon=False)
    return figure


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
    # The directions across the line, and the planes they sweep through time, which read the electric field along them.
    across = stack((mv.x.dual(), mv.y.dual()))                                        # [2] Direction-
    swept = across & -mv.t.dual()                                                     # [2] Bivector-
    # The optic axes at every third node of the slab, reaching out across the line on either side.
    rod_middles = slab[::3]                                                           # [rods] Direction-
    rod_reach = directors[::3] * (optic_half_length * field_length)                   # [rods] Direction-

    # The stretch of line framed, the box of the slab's reach across it.
    length = nodes[-1] - nodes[0]                                                     # [] Direction-
    framed = stack((nodes[0] + length * framed_start, nodes[-1] - length * framed_end))   # [ends] Direction-
    half_width = slab_half_width * field_length
    corners = framed[:, None] + (across * (half_width * square_corners)).sum(axis=-1)   # [ends, corners] Direction-
    view = solids.view([gpu_points(corners)], 0.0, pulse_elevation, pulse_azimuth, pulse_margin, n_pulse_pixels)
    line = solids.rods(gpu_points(nodes[0]), gpu_points(nodes[-1]), line_width / 2 * view.pixel)[None]
    # The slab: a box about its middle on the line, as long as the slab's stretch of it, turned upright with
    # everything else.
    middle = gpu_points((slab[0] + slab[-1]) / 2)                                     # [] Point-
    slab_length = (mv.z & (slab[-1] - slab[0])).to_array()
    box = solids.boxes(solids.translator(middle - solids.origin) * upright,
                       np.array([half_width, half_width, slab_length / 2]))[None]
    optics = solids.rods(gpu_points(rod_middles - rod_reach), gpu_points(rod_middles + rod_reach),
                         optic_width / 2 * view.pixel)

    # The line, the optic axes and the slab stay put.
    picture = Scene(view, np.ones(3), [(line, line_colour), (optics, optic_colour)], [],
                    inks=[], translucent=[(box, slab_colour, slab_opacity)])
    for planes in electric:
        readings = planes[:, None] & swept                                            # [nodes, 2] Scalar+
        # The field across the line, a direction from its readings along x and y, drawn out from each node.
        transverse = (across * readings).sum(axis=-1)                                 # [nodes] Direction-
        tips = gpu_points(nodes + transverse * field_length)                          # [nodes] Point-
        # The field's direction around the line sets its hue, and its strength how far its colour shows.
        field = readings.to_array()                                                   # [nodes, 2]
        hue = (np.arctan2(field[:, 1], field[:, 0]) / (2 * np.pi)) % 1
        strength = np.clip(np.hypot(*field.T) * 4, 0, 1)
        colours = hsv_to_rgb(np.column_stack([hue, np.full_like(hue, 0.85), np.ones_like(hue)]))
        # Every other node's spoke, where the field shows.
        spoked = np.flatnonzero(strength > spoke_floor)[::2]
        spokes = strokes.segments(gpu_points(nodes[spoked]), tips[spoked],
                                  faded(colours, spoke_opacity * strength)[spoked], spoke_width, view)
        # Each piece of the path in the colour and strength halfway along it.
        shades = faded(colours, strength)
        path = strokes.segments(tips[:-1], tips[1:], (shades[:-1] + shades[1:]) / 2, path_width, view)
        yield picture.draw([], inks=[], translucent=[], meshes=[(spokes, 1.0), (path, 1.0)])
