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
def transverse(fields: core.Bivector) -> np.ndarray:
    """Electric components across the beam for the observer's screen."""
    return np.stack(((fields | core.HORIZONTAL).to_array(),
                     (fields | core.VERTICAL).to_array()), axis=-1)          # [..., 2]


def screen_axes(axes: plt.Axes, limit: float) -> None:
    """A fixed scale on both transverse directions."""
    axes.axhline(0, color="0.85", linewidth=0.7, zorder=0)
    axes.axvline(0, color="0.85", linewidth=0.7, zorder=0)
    axes.set(xlim=(-limit, limit), ylim=(-limit, limit), aspect="equal",
             xlabel="Horizontal", ylabel="Vertical")
    axes.spines[["top", "right"]].set_visible(False)


class ResponseView:
    """One response figure for a fixed set of pumps, its artists updated for each crystal."""

    def __init__(self, pumps: core.Bivector) -> None:
        pumps = transverse(pumps)                                            # [angles, 2]
        pump = pumps[0]
        self.angles = np.arctan2(pumps[:, 1], pumps[:, 0])                  # [angles]
        self.selected_angle = np.arctan2(pump[1], pump[0])

        self.figure = plt.figure(figsize=(12.5, 4.1), layout="constrained")
        crystal_axes = self.figure.add_subplot(1, 3, 1, projection="3d")
        response_axes = self.figure.add_subplot(1, 3, 2)
        self.intensity_axes = self.figure.add_subplot(1, 3, 3, projection="polar")

        # The four segments share a centre; their endpoints show the tetrahedral geometry.
        self.bond_lines = Line3DCollection([], colors="0.45", linewidths=2)
        crystal_axes.add_collection3d(self.bond_lines, autolim=False)
        self.bond_ends = crystal_axes.scatter([], [], [], s=45, color=GENERATED_COLOUR, depthshade=False)
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
        self.harmonic_curve, = response_axes.plot([], [], color=GENERATED_COLOUR,
                                                  linewidth=2, alpha=0.65)
        response_axes.quiver(0, 0, pump[0], pump[1], color=PUMP_COLOUR,
                             angles="xy", scale_units="xy", scale=1, width=0.012,
                             label="Pump")
        self.generated_arrow = response_axes.quiver(0, 0, 0, 0, color=GENERATED_COLOUR,
                                                    angles="xy", scale_units="xy", scale=1, width=0.012,
                                                    label="Second harmonic")
        screen_axes(response_axes, 1.15)
        response_axes.set_title("Transverse polarization")
        response_axes.legend(loc="upper right", fontsize=8, frameon=False)

        self.intensity_curve, = self.intensity_axes.plot([], [], color=GENERATED_COLOUR, linewidth=2)
        self.intensity_fill, = self.intensity_axes.fill([], [], color=GENERATED_COLOUR, alpha=0.08)
        self.selected_power, = self.intensity_axes.plot([], [], "o", color=PUMP_COLOUR, markersize=7)
        self.intensity_axes.yaxis.set_major_locator(MaxNLocator(4))
        self.intensity_axes.set_title("Power by pump direction", pad=22)
        self.intensity_axes.set_thetagrids([0, 45, 90, 135, 180, 225, 270, 315])
        self.intensity_axes.set_rlabel_position(22.5)
        self.intensity_axes.tick_params(labelsize=8)

    def update(self, bonds: core.Bivector, harmonic: core.Bivector) -> plt.Figure:
        intensity = (harmonic | harmonic).to_array()                         # [angles]
        bonds = (core.mv.t | bonds).cast(core.ga.subspace("x y z")).kernel   # [bonds, 3]
        harmonic = transverse(harmonic)                                      # [angles, 2]
        self.bond_lines.set_segments(np.stack((np.zeros_like(bonds), bonds), axis=1))   # [bonds, 2, 3]
        self.bond_ends.set_offsets(bonds[:, :2])
        self.bond_ends.set_3d_properties(bonds[:, 2], "z")
        self.harmonic_curve.set_data(harmonic[:, 0], harmonic[:, 1])
        self.generated_arrow.set_UVC(*harmonic[0])
        self.intensity_curve.set_data(self.angles, intensity)
        self.intensity_fill.set_xy(np.stack((self.angles, intensity), axis=-1))
        self.selected_power.set_data([self.selected_angle], [intensity[0]])
        self.intensity_axes.set_ylim(0, 1.12 * float(np.max(intensity)))
        return self.figure


def draw_response(pumps: core.Bivector, bonds: core.Bivector, harmonic: core.Bivector) -> plt.Figure:
    """Crystal bonds, the doubled-frequency polarization over pump polarizations and its power; the
    first pump, horizontal, marked."""
    return ResponseView(pumps).update(bonds, harmonic)


def draw_waveform(phase: np.ndarray, pump: core.Bivector, harmonic: core.Bivector) -> plt.Figure:
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


def draw_mixing(pumps: core.Bivector, probes: core.Bivector, generated: core.Bivector) -> plt.Figure:
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


def draw_growth(depths: np.ndarray, amplitude: core.Bivector) -> plt.Figure:
    """The amplitude's path in the phase plane and the power along each crystal."""
    power = (amplitude | amplitude).sum(axis=-1).to_array()                  # [cases, slices]
    amplitude = (amplitude | core.VERTICAL).to_array()                      # [cases, slices, quadratures]
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


def animate(pumps: core.Bivector, frames: Iterable[tuple[core.Bivector, core.Bivector]]) -> list[np.ndarray]:
    """The response while the crystal turns about the beam, one frame per turn."""
    view = ResponseView(pumps)
    images = []
    for bonds, harmonic in frames:
        images.append(capture(view.update(bonds, harmonic)))
        # The first frame places the panels; the frames after it keep those places.
        view.figure.set_layout_engine("none")
    plt.close(view.figure)
    return images
