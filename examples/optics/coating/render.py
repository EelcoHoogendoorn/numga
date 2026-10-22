"""Rotating optic axes, polarization-selective reflection, and axion interfaces."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb, to_rgb
from matplotlib.collections import LineCollection

from examples.animation import capture
from examples.optics.coating import core
from examples.wgsl import labels, show, solids

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
        image = ax.pcolormesh(depths, times, wave, shading="nearest", cmap="RdBu_r",
                              vmin=-limit, vmax=limit, rasterized=True)
        ax.set(xlabel="depth (nm)")
    panels[0].set_ylabel("optical cycles")
    figure.colorbar(image, ax=panels, label="electric x")
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
    ax.set(xlim=(jumps[0], jumps[-1]), ylim=(0, None),
           xlabel="axion jump", ylabel="reflectance (%)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9, frameon=False)
    return figure


def pulse(nodes: core.Vector, slab: core.Vector, directors: core.Vector, electric: core.Bivector) -> list[np.ndarray]:
    """The electric field along the line at each frame, in an oblique view with x drawn up and y drawn
    aslant: the path of its tip, coloured by its direction around the axis, over faint spokes. A field
    that turns traces loops; one that does not stays in its plane. The slab is a shaded band, its optic
    axes grey rods."""
    algebra = nodes.algebra
    slant = 1000.0                                                                   # nm of depth per unit of y
    depths = nodes.cast(algebra.subspace("z")).kernel[:, 0]                           # [nodes]
    fields = electric.cast(algebra.subspace("xt yt")).kernel                         # [frames, nodes, 2]
    rod_depths = slab.cast(algebra.subspace("z")).kernel[::3, 0]
    rods = 0.6 * directors.cast(algebra.subspace("x y")).kernel[::3]                 # [rods, 2]

    def tips(at: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        return np.column_stack([at + slant * vectors[:, 1], vectors[:, 0]])

    figure = plt.figure(figsize=(9, 2.4), dpi=64, facecolor="white")
    ax = figure.add_axes((0, 0, 1, 1))
    slab_depths = slab.cast(algebra.subspace("z")).kernel[:, 0]
    band = ax.axvspan(slab_depths[0], slab_depths[-1], color="#e8edf3", zorder=0)
    axes_rods = LineCollection(np.stack([tips(rod_depths, -rods), tips(rod_depths, rods)], axis=1),
                               colors="0.55", linewidths=0.8, alpha=0.35)
    ax.add_collection(axes_rods)
    axes_rods.set_clip_path(band)
    ax.axhline(0, color="0.8", linewidth=0.6)
    spokes = LineCollection([], linewidths=0.6)
    path = LineCollection([], linewidths=1.4)
    ax.add_collection(spokes)
    ax.add_collection(path)
    ax.set(xlim=(depths[0] + 0.08 * np.ptp(depths), depths[-1] - 0.12 * np.ptp(depths)), ylim=(-1.15, 1.15))
    ax.set_axis_off()

    frames = []
    for field in fields:
        ends = tips(depths, field)
        hue = (np.arctan2(field[:, 1], field[:, 0]) / (2 * np.pi)) % 1
        strength = np.clip(np.hypot(*field.T) * 4, 0, 1)
        colours = hsv_to_rgb(np.column_stack([hue, np.full_like(hue, 0.85), np.full_like(hue, 0.85)]))
        spokes.set_segments(np.stack([np.column_stack([depths, np.zeros_like(depths)]), ends], axis=1)[::2])
        spokes.set_color(np.column_stack([colours, 0.25 * strength])[::2])
        path.set_segments(np.stack([ends[:-1], ends[1:]], axis=1))
        path.set_color(np.column_stack([colours, strength])[:-1])
        frames.append(capture(figure))
    plt.close(figure)
    return frames

