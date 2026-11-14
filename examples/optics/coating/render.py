"""Charts and ray-traced frames display matching-film spectra, cholesteric handedness bands, internal wave patterns, axion reflection, and time-domain pulse splitting."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb, to_rgb

from numga import stack
from examples.optics.coating import core
from examples.util import charts
from numga.topology import Topology
from numga.render import Scene, simplices, solids

# Spatial projection along the observer's time line onto the rest-frame slice.
space_slice = core.real.t ^ (core.Direction & -core.real.t.dual())                # Direction- <- Direction-
# Intersecting a field bivector's dual with the observer's time 1-form recovers its spatial electric direction.
electric_direction = core.Field.dual_inverse() ^ core.real.t                      # Direction- <- Field+

case_colours = ("#2563eb", "#ea580c", "#0d9488", "#7c3aed")
# Pixel dimensions of single-panel and two-panel quantity charts.
single_size, double_size = (420, 280), (640, 280)
sm = solids.mv
# Camera elevation and azimuth in radians, viewport margin fraction, and pixel width of the 3D pulse animation.
pulse_elevation, pulse_azimuth = np.radians(6.0), np.radians(30.0)
pulse_margin, n_pulse_pixels = 0.02, 600
# Display length in nm per unit electric field, trimmed line-end shares, and stroke widths in pixels.
field_length = 3000.0
trim_start, trim_end = 0.08, 0.12
path_width, spoke_width, optic_width, line_width = 2.25, 1.1, 1.5, 1.5
# Spoke opacity factor, minimum spoke strength, optic-axis half-length, slab half-width in field units, and colours.
spoke_opacity, spoke_floor = 0.5, 0.02
optic_half_length, slab_half_width, slab_opacity = 0.6, 1.0, 0.22
slab_colour = np.array(to_rgb("#b8c4d4"))
line_colour, optic_colour = np.array([0.75, 0.76, 0.78]), np.array([0.62, 0.63, 0.65])
# Half-turn about x + z placing the propagation axis z horizontal along x and transverse x vertical along z.
upright_turn = ((sm.yz + sm.xy) * (np.pi / (2 * np.sqrt(2)))).exp()               # Rotor+
# Signed transverse corner offsets of a unit square cross-section.
square_corners = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])                   # [4, 2]


# --- plumbing -------------------------------------------------------------------------
def quantity_axes(ax: plt.Axes) -> None:
    """Hiding top and right spines leaves clean horizontal and vertical quantity axes on `ax`."""
    ax.spines[["top", "right"]].set_visible(False)


@charts.style()
def spectra_figure(wavelengths: np.ndarray, reflectance: core.Scalar) -> plt.Figure:
    """Reflected power percentages across `wavelengths` (`[n_wavelengths]`) plot one curve per coating in `reflectance` (`[3, n_wavelengths] Scalar+`)."""
    figure, ax = charts.subplots(*single_size, layout="constrained")
    ax.set_prop_cycle(color=case_colours)
    for spectrum in reflectance.real().to_array():
        ax.plot(wavelengths, spectrum * 100)
    ax.set(xlim=(wavelengths[0], wavelengths[-1]), ylim=(0, None),
           xlabel="wavelength (nm)", ylabel="reflectance (%)")
    quantity_axes(ax)
    return figure


@charts.style()
def angular_figure(wavelengths: np.ndarray, angles: np.ndarray, reflectance: core.Scalar) -> plt.Figure:
    """Two side-by-side heatmaps display circular-polarization `reflectance` (`[2, n_angles, n_wavelengths] Scalar+`) across `wavelengths` and incidence `angles`."""
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
def internal_field_figure(depths: np.ndarray, fields: core.Field) -> plt.Figure:
    """Signed `x`-component waveforms of `fields` (`[2, n_layers, n_samples] Field+`) across `depths` (`[n_layers, n_samples]`) render over two optical cycles."""
    n_cycles = 2
    n_samples_per_cycle = 80
    times = np.linspace(0, n_cycles, n_cycles * n_samples_per_cycle + 1)
    phase = 2 * np.pi * times
    flat_depths = depths.reshape(-1)
    mv = fields.real().context.multivector
    # Worldsheet plane swept by x along the observer's time direction.
    along_x = mv.x.dual() & -mv.t.dual()                                          # Excitation-
    cycle_fields = (fields.reshape(fields.shape[0], -1)[:, None] * np.exp(1j * phase)[:, None]).real()  # [...] Field+
    waves = (cycle_fields & along_x).to_array()
    limit = np.max(np.abs(waves))
    figure, panels = charts.subplots(*double_size, ncols=2, sharex=True, sharey=True,
                                     layout="constrained")
    for ax, wave in zip(panels, waves):
        ax.pcolormesh(flat_depths, times, wave, shading="nearest", cmap="RdBu_r", vmin=-limit, vmax=limit,
                      rasterized=True)
        ax.set(xlabel="depth (nm)")
        quantity_axes(ax)
    panels[0].set(yticks=np.arange(n_cycles + 1))
    panels[0].set_ylabel("optical cycles")
    return figure


@charts.style()
def handedness_figure(wavelengths: np.ndarray, reflectance: core.Scalar) -> plt.Figure:
    """Side-by-side panels plot circular-polarization `reflectance` (`[2, 2, n_wavelengths] Scalar+`) across `wavelengths` for opposite crystal twists."""
    figure, axes = charts.subplots(*double_size, ncols=2, sharex=True, sharey=True,
                                   layout="constrained")
    values = reflectance.real().to_array()
    for twist_index, ax in enumerate(axes):
        ax.set_prop_cycle(color=case_colours)
        for spectrum in values[:, twist_index]:
            ax.plot(wavelengths, spectrum * 100)
        ax.set(xlim=(wavelengths[0], wavelengths[-1]), ylim=(0, 100),
               xlabel="wavelength (nm)")
        quantity_axes(ax)
    axes[0].set_ylabel("reflectance (%)")
    return figure


@charts.style()
def axion_figure(jumps: np.ndarray, reflectance: core.Scalar) -> plt.Figure:
    """Co-polarized, cross-polarized, and total `reflectance` (`[3, n_jumps] Scalar+`) plot as percentages across axion `jumps` (`[n_jumps]`)."""
    figure, ax = charts.subplots(*single_size, layout="constrained")
    ax.set_prop_cycle(color=case_colours)
    for channel in reflectance.real().to_array():
        ax.plot(jumps, channel * 100)
    ax.set(xlim=(jumps[0], jumps[-1]), ylim=(0, None), xticks=[0],
           xlabel="axion jump", ylabel="reflectance (%)")
    quantity_axes(ax)
    return figure


def faded_colours(colours: np.ndarray, strength: np.ndarray) -> np.ndarray:
    """Blending RGB `colours` (`[..., 3]`) toward white by `strength` (`[...]`) fades weak field regions into the background."""
    return 1 - strength[..., None] * (1 - colours)


def pulse_animation(
    nodes: core.Direction,
    slab: core.Direction,
    directors: core.Direction,
    case_index: int,
    states: Iterable[tuple[core.Field, core.Field]],
) -> Iterator[np.ndarray]:
    """Yields ray-traced RGB frames of case `case_index` in `states` (`([3] Field+[V], [3] Field+[E])`) travelling along `nodes` through the translucent `slab` and its `directors`."""
    mv = nodes.context.multivector
    across = stack((mv.x.dual(), mv.y.dual()))                                    # [2] Direction-
    swept = across & -mv.t.dual()                                                 # [2] Excitation-
    rod_middles = slab[::3]                                                       # [...] Direction-
    rod_reach = directors[::3] * (optic_half_length * field_length)               # [...] Direction-

    length = nodes[-1] - nodes[0]                                                 # [] Direction-
    frame_ends = stack((nodes[0] + length * trim_start, nodes[-1] - length * trim_end))  # [2] Direction-
    half_width = slab_half_width * field_length
    corners = frame_ends[:, None] + (across * (half_width * square_corners)).sum(axis=-1)  # [2, 4] Direction-
    scene_corners = solids.origin + (upright_turn >> solids.context.include_from_algebra(space_slice(corners)))
    stations = solids.origin + (upright_turn >> solids.context.include_from_algebra(space_slice(nodes)))
    view = solids.view([scene_corners], 0.0, pulse_elevation, pulse_azimuth, pulse_margin, n_pulse_pixels)
    line = solids.rods(stations[0], stations[-1], line_width / 2 * view.pixel)[None]
    middle = solids.origin + (upright_turn >> solids.context.include_from_algebra(space_slice((slab[0] + slab[-1]) / 2)))
    slab_half_length = float(((slab[-1] - slab[0]).antinorm() / 2).to_array())
    half_extents = np.array([half_width, half_width, slab_half_length])
    box = solids.boxes(solids.translator(middle - solids.origin) * upright_turn, half_extents)[None]
    rod_centres = solids.origin + (upright_turn >> solids.context.include_from_algebra(space_slice(rod_middles)))
    rod_offsets = upright_turn >> solids.context.include_from_algebra(space_slice(rod_reach))
    optics = solids.rods(rod_centres - rod_offsets, rod_centres + rod_offsets, optic_width / 2 * view.pixel)

    picture = Scene(view, np.ones(3), [(line, line_colour), (optics, optic_colour)], [],
                    inks=[], translucent=[(box, slab_colour, slab_opacity)])
    for electric, _ in states:
        planes = electric[case_index].batch()                                     # [...] Field+
        directions = electric_direction(planes)                                   # [...] Direction-
        offsets = upright_turn >> solids.context.include_from_algebra(field_length * directions)
        tips = stations + offsets                                                 # [...] Point-
        along_x, along_y = (planes[..., None] & swept).to_array().T
        hue = (np.arctan2(along_y, along_x) / (2 * np.pi)) % 1
        strength = np.clip(directions.antinorm().to_array() * 4, 0, 1)
        colours = hsv_to_rgb(np.column_stack([hue, np.full_like(hue, 0.85), np.ones_like(hue)]))
        spoke_indices = np.flatnonzero(strength > spoke_floor)[::2]
        n_spokes = len(spoke_indices)
        spoke_ends = stack((stations[spoke_indices], tips[spoke_indices]), axis=1).reshape(-1)  # [n_spokes * 2] Point-
        spokes = simplices.tubes(spoke_ends, Topology.chains([2] * n_spokes),
                                 faded_colours(colours, spoke_opacity * strength)[spoke_indices],
                                 spoke_width, view, gloss=solids.gloss)
        shades = faded_colours(colours, strength)
        path = simplices.tubes(tips, Topology.chains([len(tips)]), (shades[:-1] + shades[1:]) / 2, path_width,
                               view, gloss=solids.gloss)
        yield picture.draw([], inks=[], translucent=[], meshes=[(spokes, 1.0), (path, 1.0)])
