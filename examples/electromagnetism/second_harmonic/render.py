"""Polarization response, mixed fields and coherent growth drawn from geometric states."""

from __future__ import annotations

from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from examples.animation import capture
from examples.electromagnetism.second_harmonic import core

PUMP_COLOUR = "#3275a8"
GENERATED_COLOUR = "#d16a30"
CASE_COLOURS = ("#3275a8", "#d16a30", "#7b62a3")
# The crystals of the phase-matching scene, in its order.
CASES = ("matched", "mismatched", "mismatched, flipped every half turn of slip")


# --- plumbing -------------------------------------------------------------------------
def transverse(vectors: core.Vector) -> np.ndarray:
    """Horizontal and vertical components in the plane perpendicular to the beam."""
    return np.stack(((vectors | core.HORIZONTAL).to_array(),
                     (vectors | core.VERTICAL).to_array()), axis=-1)          # [..., 2]


def screen_axes(axes: plt.Axes, limit: float) -> None:
    """A fixed scale on both transverse directions."""
    axes.axhline(0, color="0.85", linewidth=0.7, zorder=0)
    axes.axvline(0, color="0.85", linewidth=0.7, zorder=0)
    axes.set(xlim=(-limit, limit), ylim=(-limit, limit), aspect="equal",
             xlabel="Horizontal", ylabel="Vertical")
    axes.spines[["top", "right"]].set_visible(False)


def draw_response(pumps: core.Vector, bonds: core.Vector, harmonic: core.Vector) -> plt.Figure:
    """Crystal bonds, the doubled-frequency polarization over pump polarizations and its power; the
    first pump, horizontal, marked."""
    intensity = (harmonic | harmonic).to_array()                             # [angles]
    bonds = bonds.cast(core.ga.subspace("x y z")).kernel                     # [4, 3]
    pumps = transverse(pumps)                                                # [angles, 2]
    harmonic = transverse(harmonic)                                          # [angles, 2]
    pump, generated, power = pumps[0], harmonic[0], intensity[0]
    angles = np.arctan2(pumps[:, 1], pumps[:, 0])
    selected_angle = np.arctan2(pump[1], pump[0])

    figure = plt.figure(figsize=(12.5, 4.1), layout="constrained")
    crystal_axes = figure.add_subplot(1, 3, 1, projection="3d")
    response_axes = figure.add_subplot(1, 3, 2)
    intensity_axes = figure.add_subplot(1, 3, 3, projection="polar")

    # The four segments share a centre; their endpoints show the tetrahedral geometry.
    segments = np.stack((np.zeros_like(bonds), bonds), axis=1)                 # [4, 2, 3]
    crystal_axes.add_collection3d(Line3DCollection(segments, colors="0.45", linewidths=2))
    crystal_axes.scatter(*bonds.T, s=45, color=GENERATED_COLOUR, depthshade=False)
    crystal_axes.scatter(0, 0, 0, s=30, color="0.3", depthshade=False)
    crystal_axes.set(xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1),
                     xlabel="x", ylabel="y", zlabel="z", title="Crystal bonds")
    crystal_axes.set_box_aspect((1, 1, 1))
    crystal_axes.view_init(elev=23, azim=-56)
    crystal_axes.set_xticks([-1, 0, 1])
    crystal_axes.set_yticks([-1, 0, 1])
    crystal_axes.set_zticks([-1, 0, 1])

    response_axes.plot(pumps[:, 0], pumps[:, 1], color=PUMP_COLOUR,
                       linewidth=1, alpha=0.3)
    response_axes.plot(harmonic[:, 0], harmonic[:, 1], color=GENERATED_COLOUR,
                       linewidth=2, alpha=0.65)
    response_axes.quiver(0, 0, pump[0], pump[1], color=PUMP_COLOUR,
                         angles="xy", scale_units="xy", scale=1, width=0.012,
                         label="Pump")
    response_axes.quiver(0, 0, generated[0], generated[1], color=GENERATED_COLOUR,
                         angles="xy", scale_units="xy", scale=1, width=0.012,
                         label="Second harmonic")
    screen_axes(response_axes, 1.15)
    response_axes.set_title("Transverse polarization")
    response_axes.legend(loc="upper right", fontsize=8, frameon=False)

    intensity_axes.plot(angles, intensity, color=GENERATED_COLOUR, linewidth=2)
    intensity_axes.fill(angles, intensity, color=GENERATED_COLOUR, alpha=0.08)
    intensity_axes.plot([selected_angle], [power], "o", color=PUMP_COLOUR, markersize=7)
    intensity_axes.set_ylim(0, 1.12 * float(np.max(intensity)))
    intensity_axes.yaxis.set_major_locator(MaxNLocator(4))
    intensity_axes.set_title("Power by pump direction", pad=22)
    intensity_axes.set_thetagrids([0, 45, 90, 135, 180, 225, 270, 315])
    intensity_axes.set_rlabel_position(22.5)
    intensity_axes.tick_params(labelsize=8)
    return figure


def draw_waveform(phase: np.ndarray, pump: core.Vector, harmonic: core.Vector) -> plt.Figure:
    """Signed pump and doubled-frequency field components over two pump periods."""
    cycles = phase / (2 * np.pi)                                               # [times]
    pump_field = (pump | core.HORIZONTAL).to_array()                           # [times]
    harmonic_field = (harmonic | core.VERTICAL).to_array()                     # [times]
    figure, axes = plt.subplots(figsize=(8, 3.3), layout="constrained")
    axes.axhline(0, color="0.85", linewidth=0.7, zorder=0)
    axes.plot(cycles, pump_field, color=PUMP_COLOUR, linewidth=2, label="Pump field")
    axes.plot(cycles, harmonic_field, color=GENERATED_COLOUR, linewidth=2,
              label="Second-harmonic polarization")
    axes.set(xlabel="Pump periods", ylabel="Field component", xlim=(0, 2))
    axes.spines[["top", "right"]].set_visible(False)
    axes.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2,
                fontsize=9, frameon=False)
    return figure


def draw_mixing(pumps: core.Vector, probes: core.Vector, generated: core.Vector) -> plt.Figure:
    """A probe circle and its image with each fixed pump, at a shared display scale."""
    pumps = transverse(pumps)                                                # [cases, 2]
    probes = transverse(probes)                                              # [angles, 2]
    generated = transverse(generated)                                        # [cases, angles, 2]
    limit = 1.25 * max(float(np.max(np.abs(probes))), float(np.max(np.abs(generated))))
    figure, grid = plt.subplots(1, len(pumps), figsize=(4 * len(pumps), 4),
                                squeeze=False, layout="constrained")
    for axes, pump, response in zip(grid[0], pumps, generated):
        # The pale arrow indicates the fixed pump direction at a display-only length.
        axes.quiver(0, 0, 0.85 * limit * pump[0], 0.85 * limit * pump[1],
                    angles="xy", scale_units="xy", scale=1, color="0.55",
                    alpha=0.4, width=0.009, label="Fixed pump")
        axes.plot(probes[:, 0], probes[:, 1], color=PUMP_COLOUR, linewidth=1.8,
                  linestyle="--", zorder=3, label="Probe")
        axes.plot(response[:, 0], response[:, 1], color=GENERATED_COLOUR,
                  linewidth=2, label="Mixed response")
        screen_axes(axes, limit)
        angle = np.degrees(np.arctan2(pump[1], pump[0]))
        axes.set_title(f"Pump at {angle:.0f}°")
    figure.legend(*grid[0, 0].get_legend_handles_labels(), loc="outside lower center",
                  ncol=3, fontsize=9, frameon=False)
    return figure


def draw_growth(depths: np.ndarray, amplitude: core.Phasor) -> plt.Figure:
    """The amplitude's path in the phase plane and the power along each crystal."""
    power = amplitude.symmetric_reverse_product().to_array()                 # [cases, slices]
    amplitude = amplitude.cast(core.ga.subspace("1 xyz")).kernel             # [cases, slices, 2]
    figure, axes = plt.subplots(1, 2, figsize=(9.5, 4), layout="constrained")
    phasor_axes, power_axes = axes
    for label, path, intensity, colour in zip(CASES, amplitude, power, CASE_COLOURS):
        phasor_axes.plot(path[:, 0], path[:, 1], color=colour, linewidth=2, label=label)
        phasor_axes.plot(path[-1, 0], path[-1, 1], "o", color=colour, markersize=5)
        power_axes.plot(depths, intensity, color=colour, linewidth=2, label=label)
    phasor_axes.axhline(0, color="0.85", linewidth=0.7, zorder=0)
    phasor_axes.axvline(0, color="0.85", linewidth=0.7, zorder=0)
    phasor_axes.set(aspect="equal", xlabel="in phase", ylabel="in quadrature", title="accumulated field")
    power_axes.set(xlabel="depth in the crystal", ylabel="generated power", title="growth", xlim=(0, 1))
    for panel in axes:
        panel.spines[["top", "right"]].set_visible(False)
    power_axes.legend(loc="upper left", frameon=False, fontsize=9)
    return figure


def animate(pumps: core.Vector, frames: Iterable[tuple[core.Vector, core.Vector]]) -> list[np.ndarray]:
    """The response while the crystal turns about the beam, one frame per turn."""
    images = []
    for bonds, harmonic in frames:
        figure = draw_response(pumps, bonds, harmonic)
        images.append(capture(figure))
        plt.close(figure)
    return images
