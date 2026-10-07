"""Rotating optic axes, polarization-selective reflection, and axion interfaces."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from examples.animation import capture
from examples.optics.coating import core

COLOURS = ("#2563eb", "#ea580c", "#0d9488", "#7c3aed")


# --- plumbing -------------------------------------------------------------------------
def spectra(wavelengths: np.ndarray, reflectance: core.Scalar, names: tuple[str, ...]) -> plt.Figure:
    """Reflected power as a percentage of the incident power for each coating."""
    figure, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    ax.set_prop_cycle(color=COLOURS)
    values = reflectance.real().cast(reflectance.algebra.subspace.scalar()).kernel[..., 0]
    for spectrum, name in zip(values, names):
        ax.plot(wavelengths, spectrum * 100, linewidth=1.8, label=name)
    ax.set(xlim=(wavelengths[0], wavelengths[-1]), ylim=(0, None),
           xlabel="wavelength (nm)", ylabel="reflectance (%)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9, frameon=False)
    return figure


def angular(
    wavelengths: np.ndarray, angles: np.ndarray, reflectance: core.Scalar, names: tuple[str, ...],
) -> plt.Figure:
    """The two polarizations over wavelength and incidence angle, on one reflectance scale."""
    figure, panels = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True, sharey=True,
                                 layout="constrained")
    values = reflectance.real().cast(reflectance.algebra.subspace.scalar()).kernel[..., 0]
    for ax, spectrum, name in zip(panels, values, names):
        image = ax.pcolormesh(wavelengths, np.degrees(angles), spectrum,
                              shading="nearest", cmap="magma", vmin=0, vmax=1, rasterized=True)
        ax.set(xlabel="wavelength (nm)", title=name)
    panels[0].set_ylabel("incidence angle (degrees)")
    figure.colorbar(image, ax=panels, label="reflectance")
    return figure


def field(depths: np.ndarray, electric: core.Vector, names: tuple[str, ...]) -> plt.Figure:
    """The signed electric x component through depth over two optical cycles."""
    cycles = 2
    samples_per_cycle = 80
    times = np.linspace(0, cycles, cycles * samples_per_cycle + 1)
    phase = 2 * np.pi * times
    in_phase = electric.real().cast(electric.algebra.subspace("x")).kernel[..., 0]
    quadrature = (-1j * electric).real().cast(electric.algebra.subspace("x")).kernel[..., 0]
    waves = (in_phase[:, None, :] * np.cos(phase)[None, :, None]
             - quadrature[:, None, :] * np.sin(phase)[None, :, None])
    limit = np.max(np.abs(waves))
    figure, panels = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True, sharey=True,
                                 layout="constrained")
    for ax, wave, name in zip(panels, waves, names):
        image = ax.pcolormesh(depths, times, wave, shading="gouraud", cmap="RdBu_r",
                              vmin=-limit, vmax=limit, rasterized=True)
        ax.set(xlabel="depth (nm)", title=name)
    panels[0].set_ylabel("optical cycles")
    figure.colorbar(image, ax=panels, label="electric x")
    return figure


def twist(centres: core.Vector, directors: core.Vector) -> plt.Figure:
    """Undirected optic axes at the layer depths; rod lengths are schematic."""
    figure = plt.figure(figsize=(5, 6), layout="constrained")
    ax = figure.add_subplot(111, projection="3d")
    positions = centres.real().cast(centres.algebra.subspace("x y z")).kernel
    directions = directors.real().cast(directors.algebra.subspace("x y z")).kernel
    half_width = np.ptp(positions[:, 2]) * 0.14
    ends = np.stack([positions - half_width * directions, positions + half_width * directions], axis=1)
    ax.add_collection3d(Line3DCollection(ends, colors=COLOURS[0], linewidths=1.4))
    ax.plot(*positions.T, color="#cbd5e1", linewidth=0.7)
    ax.set(xlim=(-half_width * 1.15, half_width * 1.15),
           ylim=(-half_width * 1.15, half_width * 1.15),
           zlim=(positions[:, 2].min(), positions[:, 2].max()),
           xticks=[], yticks=[], zlabel="depth (nm)")
    ax.set_box_aspect((1, 1, 2.4))
    ax.set_proj_type("ortho")
    ax.view_init(elev=16, azim=-55)
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.line.set_color("0.8")
    return figure


def handedness(
    wavelengths: np.ndarray, reflectance: core.Scalar,
    names: tuple[str, ...], polarizations: tuple[str, ...],
) -> plt.Figure:
    """Opposite material twists, with the same colours for the incident polarizations."""
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True, sharey=True,
                               layout="constrained")
    values = reflectance.real().cast(reflectance.algebra.subspace.scalar()).kernel[..., 0]
    for ax, spectra, name in zip(axes, values, names):
        ax.set_prop_cycle(color=COLOURS)
        for spectrum, polarization in zip(spectra, polarizations):
            ax.plot(wavelengths, spectrum * 100, linewidth=1.8, label=polarization)
        ax.set(xlim=(wavelengths[0], wavelengths[-1]), ylim=(0, 100),
               xlabel="wavelength (nm)", title=name)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("reflectance (%)")
    axes[1].legend(fontsize=9, frameon=False)
    return figure


def axion(jumps: np.ndarray, reflectance: core.Scalar, names: tuple[str, ...]) -> plt.Figure:
    """Reflected power in each polarization channel across an axion discontinuity."""
    figure, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    ax.set_prop_cycle(color=COLOURS)
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


def inline(frames: list[np.ndarray], duration_ms: int):
    """Frames as a looping GIF to show in a notebook, kept in memory."""
    from io import BytesIO
    from IPython.display import Image as Shown
    from PIL import Image
    images = [Image.fromarray(pixels) for pixels in frames]
    buffer = BytesIO()
    images[0].save(buffer, format="GIF", save_all=True, append_images=images[1:], duration=duration_ms, loop=0)
    return Shown(data=buffer.getvalue(), format="gif")
