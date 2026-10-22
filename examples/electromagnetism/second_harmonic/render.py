"""Polarization response, mixed fields and coherent growth drawn from geometric states."""

from __future__ import annotations

from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.ticker import MaxNLocator

from numga import stack

from examples.animation import capture
from examples.electromagnetism.second_harmonic import core
from examples.wgsl import labels, show, solids

PUMP_COLOUR = "#3275a8"
GENERATED_COLOUR = "#d16a30"
CASE_COLOURS = ("#3275a8", "#d16a30", "#7b62a3")
# The crystals of the phase-matching scene, in its order.
CASES = ("matched", "mismatched", "mismatched, flipped every half turn of slip")
sm = solids.mv
BOND_COLOUR, CENTRE_COLOUR, AXIS_COLOUR = np.full(3, 0.45), np.full(3, 0.3), np.array([0.68, 0.70, 0.75])
LABEL_COLOUR = np.array([0.4, 0.42, 0.47])
# The crystal's view: elevation and azimuth in radians, azimuth zero looking along +y; the half width of the square
# view about the bonds' centre, and the pixels across it.
ELEVATION, AZIMUTH = np.radians(23.0), np.radians(34.0)
HALF_WIDTH, N_PIXELS = 1.35, 400
# The widths of the bonds and the axes' arrows, the arrowheads' length and width, the radii of the bonds' ends and
# their centre, in pixels; how far the axes reach and their names lie, in bond lengths.
BOND_WIDTH, ARROW_WIDTH, HEAD_LENGTH, HEAD_WIDTH, END_RADIUS, CENTRE_RADIUS = 5.0, 3.0, 12.0, 7.0, 7.0, 5.0
AXIS_LENGTH, LABEL_DISTANCE = 1.15, 1.27


# --- plumbing -------------------------------------------------------------------------
def transverse(fields: core.Field) -> np.ndarray:
    """Horizontal and vertical components of fields across the beam, for the observer's screen: their
    pairings with the polarizations along those directions."""
    polarizations = stack((core.HORIZONTAL_POLARIZATION, core.VERTICAL_POLARIZATION))
    return (fields[..., None] & polarizations).to_array()                     # [..., 2]


def transverse_polarization(polarizations: core.Polarization) -> np.ndarray:
    """Horizontal and vertical components of polarizations across the beam: the pairings of the
    horizontal and vertical fields with them."""
    fields = stack((core.HORIZONTAL_FIELD, core.VERTICAL_FIELD))
    return (fields & polarizations[..., None]).to_array()                     # [..., 2]


def coordinates(directions: core.Direction) -> np.ndarray:
    """x, y and z of directions in space: their pairings with the planes x, y and z."""
    return (stack((core.mv.x, core.mv.y, core.mv.z)) & directions[..., None]).to_array()   # [..., 3]


def screen_axes(axes: plt.Axes, limit: float) -> None:
    """A fixed scale on both transverse directions."""
    axes.axhline(0, color="0.85", linewidth=0.7, zorder=0)
    axes.axvline(0, color="0.85", linewidth=0.7, zorder=0)
    axes.set(xlim=(-limit, limit), ylim=(-limit, limit), aspect="equal",
             xlabel="Horizontal", ylabel="Vertical")
    axes.spines[["top", "right"]].set_visible(False)


def crystal_image(bonds: np.ndarray, view: solids.View) -> np.ndarray:
    """The bonds `[bonds, 3]` as rods from their shared centre to their ends, the tetrahedral geometry, among the axes
    of space."""
    ends = sm.point(bonds)                                                   # [bonds] Point
    shafts, heads = solids.arrows(sm.point(-np.eye(3)), sm.point(AXIS_LENGTH * np.eye(3)), ARROW_WIDTH / 2 * view.pixel,
                                  HEAD_LENGTH * view.pixel, HEAD_WIDTH / 2 * view.pixel)
    parts = [(shafts, AXIS_COLOUR), (heads, AXIS_COLOUR),
             (solids.rods(solids.origin, ends, BOND_WIDTH / 2 * view.pixel), BOND_COLOUR),
             (solids.ball(ends, END_RADIUS * view.pixel)[:, None], np.array(to_rgb(GENERATED_COLOUR))),
             (solids.ball(solids.origin, CENTRE_RADIUS * view.pixel)[None, None], CENTRE_COLOUR)]
    return solids.draw(parts, view, np.ones(3))


class ResponseView:
    """One response figure for a fixed set of pumps, its artists updated for each crystal."""

    def __init__(self, pumps: core.Field) -> None:
        pumps = transverse(pumps)                                            # [angles, 2]
        pump = pumps[0]
        self.angles = np.arctan2(pumps[:, 1], pumps[:, 0])                  # [angles]
        self.selected_angle = np.arctan2(pump[1], pump[0])

        self.figure = plt.figure(figsize=(12.5, 4.1), layout="constrained")
        crystal_axes = self.figure.add_subplot(1, 3, 1)
        response_axes = self.figure.add_subplot(1, 3, 2)
        self.intensity_axes = self.figure.add_subplot(1, 3, 3, projection="polar")

        # The crystal's image, blank until the bonds are drawn, with each axis named just past its arrow's tip.
        self.view = solids.View(solids.aim(solids.origin, ELEVATION, AZIMUTH), np.full(2, HALF_WIDTH), N_PIXELS)
        self.crystal = show(crystal_axes, np.full((*self.view.shape, 3), 255, dtype=np.uint8), self.view)
        labels(crystal_axes, self.view, sm.point(LABEL_DISTANCE * np.eye(3)), list("xyz"), ha="center", va="center",
               fontsize=11, color=LABEL_COLOUR)

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
        response_axes.legend(loc="upper right", fontsize=8, frameon=False)

        self.intensity_curve, = self.intensity_axes.plot([], [], color=GENERATED_COLOUR, linewidth=2)
        self.intensity_fill, = self.intensity_axes.fill([], [], color=GENERATED_COLOUR, alpha=0.08)
        self.selected_power, = self.intensity_axes.plot([], [], "o", color=PUMP_COLOUR, markersize=7)
        self.intensity_axes.yaxis.set_major_locator(MaxNLocator(4))
        self.intensity_axes.set_thetagrids([0, 45, 90, 135, 180, 225, 270, 315])
        self.intensity_axes.set_rlabel_position(22.5)
        self.intensity_axes.tick_params(labelsize=8)

    def update(self, bonds: core.Direction, harmonic: core.Polarization) -> plt.Figure:
        bonds = coordinates(bonds)                                           # [bonds, 3]
        harmonic = transverse_polarization(harmonic)                         # [angles, 2]
        intensity = (harmonic ** 2).sum(axis=-1)                             # [angles]
        self.crystal.set_data(crystal_image(bonds, self.view))
        self.harmonic_curve.set_data(harmonic[:, 0], harmonic[:, 1])
        self.generated_arrow.set_UVC(*harmonic[0])
        self.intensity_curve.set_data(self.angles, intensity)
        self.intensity_fill.set_xy(np.stack((self.angles, intensity), axis=-1))
        self.selected_power.set_data([self.selected_angle], [intensity[0]])
        self.intensity_axes.set_ylim(0, 1.12 * float(np.max(intensity)))
        return self.figure


def draw_response(pumps: core.Field, bonds: core.Direction, harmonic: core.Polarization) -> plt.Figure:
    """Crystal bonds, the doubled-frequency polarization over pump polarizations and its power; the
    first pump, horizontal, marked."""
    return ResponseView(pumps).update(bonds, harmonic)


def draw_waveform(phase: np.ndarray, pump: core.Field, harmonic: core.Polarization) -> plt.Figure:
    """Signed pump and doubled-frequency field components over two pump periods."""
    cycles = phase / (2 * np.pi)                                               # [times]
    pump_field = transverse(pump)[..., 0]                                     # [times]
    harmonic_field = transverse_polarization(harmonic)[..., 1]                # [times]
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


def draw_mixing(pumps: core.Field, probes: core.Field, generated: core.Polarization) -> plt.Figure:
    """A probe circle and its image with each fixed pump, at a shared display scale."""
    pumps = transverse(pumps)                                                # [cases, 2]
    probes = transverse(probes)                                              # [angles, 2]
    generated = transverse_polarization(generated)                           # [cases, angles, 2]
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
    figure.legend(*grid[0, 0].get_legend_handles_labels(), loc="outside lower center",
                  ncol=3, fontsize=9, frameon=False)
    return figure


def draw_growth(depths: np.ndarray, amplitude: core.Polarization) -> plt.Figure:
    """The amplitude's vertical path in the phase plane and the power along each crystal."""
    components = transverse_polarization(amplitude)                          # [cases, slices, quadratures, 2]
    power = (components ** 2).sum(axis=(-2, -1))                            # [cases, slices]
    amplitude = components[..., 1]                                          # [cases, slices, quadratures]
    figure, axes = plt.subplots(1, 2, figsize=(9.5, 4), layout="constrained")
    phasor_axes, power_axes = axes
    for label, path, intensity, colour in zip(CASES, amplitude, power, CASE_COLOURS):
        phasor_axes.plot(path[:, 0], path[:, 1], color=colour, linewidth=2, label=label)
        phasor_axes.plot(path[-1, 0], path[-1, 1], "o", color=colour, markersize=5)
        power_axes.plot(depths, intensity, color=colour, linewidth=2, label=label)
    phasor_axes.axhline(0, color="0.85", linewidth=0.7, zorder=0)
    phasor_axes.axvline(0, color="0.85", linewidth=0.7, zorder=0)
    phasor_axes.set(aspect="equal", xlabel="in phase", ylabel="in quadrature")
    power_axes.set(xlabel="depth in the crystal", ylabel="generated power", xlim=(0, 1))
    for panel in axes:
        panel.spines[["top", "right"]].set_visible(False)
    power_axes.legend(loc="upper left", frameon=False, fontsize=9)
    return figure


def animate(pumps: core.Field, frames: Iterable[tuple[core.Direction, core.Polarization]]) -> list[np.ndarray]:
    """The response while the crystal turns about the beam, one frame per turn."""
    view = ResponseView(pumps)
    images = []
    for bonds, harmonic in frames:
        images.append(capture(view.update(bonds, harmonic)))
        # The first frame places the panels; the frames after it keep those places.
        view.figure.set_layout_engine("none")
    plt.close(view.figure)
    return images
