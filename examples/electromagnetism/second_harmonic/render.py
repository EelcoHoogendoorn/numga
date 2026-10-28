"""Polarization response, mixed fields and coherent growth drawn from geometric states."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.ticker import MaxNLocator

from numga import stack

from examples.util import charts
from examples.util.animation import capture
from examples.electromagnetism.second_harmonic import core
from examples.util.wgsl import fitted, labels, show, solids
from examples.util.wgsl.scene import Scene

pump_colour = "#3275a8"
generated_colour = "#d16a30"
case_colours = ("#3275a8", "#d16a30", "#7b62a3")
# The crystals of the phase-matching scene, in its order.
cases = ("matched", "mismatched", "mismatched, flipped every half turn of slip")
sm = solids.mv
bond_colour, centre_colour, axis_colour = np.full(3, 0.45), np.full(3, 0.3), np.array([0.68, 0.70, 0.75])
label_colour = np.array([0.4, 0.42, 0.47])
# The charts' sizes in pixels: the response, the crystal beside the response and its power, all of the column; the
# waveform, two curves over two periods; the mixing, a panel to a pump above the legend; the phase matching, the
# phase plane beside the power along the crystals.
response_size, waveform_size, growth_size = (800, 280), (460, 220), (640, 300)
mixing_panel_size = (260, 290)
# The crystal's view: elevation and azimuth in radians, azimuth zero looking along +y, and the half width of the
# square view about the bonds' centre.
elevation, azimuth = np.radians(23.0), np.radians(34.0)
half_width = 1.35
# The widths of the bonds and the axes' arrows, the arrowheads' length and width, the radii of the bonds' ends and
# their centre, in pixels; how far the axes reach and their names lie, in bond lengths.
bond_width, arrow_width, head_length, head_width, end_radius, centre_radius = 3.2, 1.9, 7.7, 4.5, 4.5, 3.2
axis_length, label_distance = 1.15, 1.27
# The beam along the crystal's face diagonal, x + y, coming towards the screen's viewer, across which the screen's
# directions are y - x, to the right, and z, up; how far the beam reaches either side of the crystal, in bond lengths.
beam, beam_reach = np.array([1.0, 1.0, 0.0]) / np.sqrt(2), 1.25


# --- plumbing -------------------------------------------------------------------------
def transverse(fields: core.Field) -> np.ndarray:
    """Horizontal and vertical components of fields across the beam, for the observer's screen: their
    pairings with the polarizations along those directions."""
    polarizations = stack((core.horizontal_polarization, core.vertical_polarization))
    return (fields[..., None] & polarizations).to_array()                     # [..., 2]


def transverse_polarization(polarizations: core.Polarization) -> np.ndarray:
    """Horizontal and vertical components of polarizations across the beam: the pairings of the
    horizontal and vertical fields with them."""
    fields = stack((core.horizontal_field, core.vertical_field))
    return (fields & polarizations[..., None]).to_array()                     # [..., 2]


def coordinates(directions: core.Direction) -> np.ndarray:
    """x, y and z of directions in space: their pairings with the planes x, y and z."""
    mv = directions.context.multivector
    return (stack((mv.x, mv.y, mv.z)) & directions[..., None]).to_array()   # [..., 3]


def cross(axes: plt.Axes, extents: np.ndarray, across: str, up: str) -> None:
    """A plane within extents `[2, 2]`, lowest and highest across and up, without scales: a faint cross through the
    origin, its lines named just past their ends, clear of what the plane holds."""
    axes.axhline(0, color="0.85", linewidth=charts.FRAME_WIDTH, zorder=0)
    axes.axvline(0, color="0.85", linewidth=charts.FRAME_WIDTH, zorder=0)
    axes.text(extents[0, 1], 0, f" {across}", ha="left", va="center", color=label_colour)
    axes.text(0, extents[1, 1], up, ha="center", va="bottom", color=label_colour)
    axes.set(xlim=extents[0], ylim=extents[1], aspect="equal")
    axes.set_axis_off()


def screen_axes(axes: plt.Axes, limit: float) -> None:
    """The plane across the beam, one extent on both directions, named by the directions of space along it."""
    cross(axes, np.array([[-limit, limit], [-limit, limit]]), "$y - x$", "$z$")


def space_scene(view: solids.View) -> Scene:
    """The frame of space, its centre and the beam through it, for the bonds to be drawn among."""
    shafts, heads = solids.frame(axis_length, arrow_width / 2 * view.pixel, head_length * view.pixel,
                                 head_width / 2 * view.pixel)
    beam_shaft, beam_head = solids.arrows(sm.point(-beam_reach * beam), sm.point(beam_reach * beam),
                                          arrow_width / 2 * view.pixel, head_length * view.pixel, head_width / 2 * view.pixel)
    beam_colour = np.array(to_rgb(pump_colour))
    parts = [(shafts, axis_colour), (heads, axis_colour), (beam_shaft[None], beam_colour), (beam_head[None], beam_colour),
             (solids.ball(solids.origin, centre_radius * view.pixel)[None, None], centre_colour)]
    return Scene(view, np.ones(3), parts, [], inks=[], translucent=[])


def crystal_image(bonds: np.ndarray, scene: Scene) -> np.ndarray:
    """The bonds `[bonds, 3]` as rods from their shared centre to their ends, the tetrahedral geometry, among the
    scene's frame of space and beam."""
    ends = sm.point(bonds)                                                   # [bonds] Point-
    view = scene.view
    return scene.draw([(solids.rods(solids.origin, ends, bond_width / 2 * view.pixel), bond_colour),
                       (solids.ball(ends, end_radius * view.pixel)[:, None], np.array(to_rgb(generated_colour)))],
                       inks=[], translucent=[], meshes=[])


class ResponseView:
    """One response figure for a fixed set of pumps, its artists updated for each crystal."""

    def __init__(self, pumps: core.Field) -> None:
        pumps = transverse(pumps)                                            # [angles, 2]
        pump = pumps[0]
        self.angles = np.arctan2(pumps[:, 1], pumps[:, 0])                  # [angles]
        self.selected_angle = np.arctan2(pump[1], pump[0])

        width, height = response_size
        self.figure = plt.figure(figsize=(width / charts.DPI, height / charts.DPI), dpi=charts.DPI,
                                 layout="constrained")
        crystal_axes = self.figure.add_subplot(1, 3, 1)
        response_axes = self.figure.add_subplot(1, 3, 2)
        self.intensity_axes = self.figure.add_subplot(1, 3, 3, projection="polar")

        response_axes.plot(pumps[:, 0], pumps[:, 1], color=pump_colour, alpha=0.3)
        self.harmonic_curve, = response_axes.plot([], [], color=generated_colour, alpha=0.65)
        response_axes.quiver(0, 0, pump[0], pump[1], color=pump_colour,
                             angles="xy", scale_units="xy", scale=1, width=0.012,
                             label="Pump")
        self.generated_arrow = response_axes.quiver(0, 0, 0, 0, color=generated_colour,
                                                    angles="xy", scale_units="xy", scale=1, width=0.012,
                                                    label="Second harmonic")
        screen_axes(response_axes, 1.15)
        response_axes.legend(loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=2, frameon=False)

        self.intensity_curve, = self.intensity_axes.plot([], [], color=generated_colour)
        self.intensity_fill, = self.intensity_axes.fill([], [], color=generated_colour, alpha=0.08)
        self.selected_power, = self.intensity_axes.plot([], [], "o", color=pump_colour)
        self.intensity_axes.yaxis.set_major_locator(MaxNLocator(4))
        self.intensity_axes.set_yticklabels([])
        self.intensity_axes.set_thetagrids([0, 45, 90, 135, 180, 225, 270, 315])
        self.intensity_axes.set_rlabel_position(22.5)

        # The figure laid out once, the crystal's axes as bare as its image leaves them, and kept so: at every
        # resolution it is drawn at, the panels stay put and the crystal's image fills its room.
        crystal_axes.set_axis_off()
        self.figure.get_layout_engine().execute(self.figure)
        self.figure.set_layout_engine("none")
        # The crystal's image, blank until the bonds are drawn, as many pixels across as its room holds, with each
        # axis named just past its arrow's tip.
        self.view = fitted(solids.View(solids.aim(solids.origin, elevation, azimuth), np.full(2, half_width), 1),
                           crystal_axes)
        self.crystal = show(crystal_axes, np.full((*self.view.shape, 3), 255, dtype=np.uint8), self.view)
        self.scene = space_scene(self.view)
        labels(crystal_axes, self.view, sm.point(np.concatenate([label_distance * np.eye(3), [beam_reach * 1.12 * beam]])),
               ["x", "y", "z", "beam"], ha="center", va="center", color=label_colour)

    def update(self, bonds: core.Direction, harmonic: core.Polarization) -> plt.Figure:
        bonds = coordinates(bonds)                                           # [bonds, 3]
        harmonic = transverse_polarization(harmonic)                         # [angles, 2]
        intensity = (harmonic ** 2).sum(axis=-1)                             # [angles]
        self.crystal.set_data(crystal_image(bonds, self.scene))
        self.harmonic_curve.set_data(harmonic[:, 0], harmonic[:, 1])
        self.generated_arrow.set_UVC(*harmonic[0])
        self.intensity_curve.set_data(self.angles, intensity)
        self.intensity_fill.set_xy(np.stack((self.angles, intensity), axis=-1))
        self.selected_power.set_data([self.selected_angle], [intensity[0]])
        self.intensity_axes.set_ylim(0, 1.12 * float(np.max(intensity)))
        return self.figure


@charts.style()
def draw_response(pumps: core.Field, bonds: core.Direction, harmonic: core.Polarization) -> plt.Figure:
    """Crystal bonds, the doubled-frequency polarization over pump polarizations and its power; the
    first pump, horizontal, marked."""
    return ResponseView(pumps).update(bonds, harmonic)


@charts.style()
def draw_waveform(phase: np.ndarray, pump: core.Field, harmonic: core.Polarization) -> plt.Figure:
    """Signed pump and doubled-frequency field components over two pump periods."""
    cycles = phase / (2 * np.pi)                                               # [times]
    pump_field = transverse(pump)[..., 0]                                     # [times]
    harmonic_field = transverse_polarization(harmonic)[..., 1]                # [times]
    figure, axes = charts.subplots(*waveform_size, layout="constrained")
    axes.axhline(0, color="0.85", linewidth=charts.FRAME_WIDTH, zorder=0)
    axes.plot(cycles, pump_field, color=pump_colour, label="Pump field")
    axes.plot(cycles, harmonic_field, color=generated_colour, label="Second-harmonic polarization")
    axes.set(xlabel="Pump periods", ylabel="Field component", xlim=(0, 2), xticks=[0, 1, 2], yticks=[])
    axes.spines[["top", "right"]].set_visible(False)
    axes.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False)
    return figure


@charts.style()
def draw_mixing(pumps: core.Field, probes: core.Field, generated: core.Polarization) -> plt.Figure:
    """A probe circle and its image with each fixed pump, at a shared display scale."""
    pumps = transverse(pumps)                                                # [cases, 2]
    probes = transverse(probes)                                              # [angles, 2]
    generated = transverse_polarization(generated)                           # [cases, angles, 2]
    limit = 1.25 * max(float(np.max(np.abs(probes))), float(np.max(np.abs(generated))))
    figure, grid = charts.subplots(len(pumps) * mixing_panel_size[0], mixing_panel_size[1], ncols=len(pumps),
                                   squeeze=False, layout="constrained")
    for axes, pump, response in zip(grid[0], pumps, generated):
        # The pale arrow indicates the fixed pump direction at a display-only length.
        axes.quiver(0, 0, 0.85 * limit * pump[0], 0.85 * limit * pump[1],
                    angles="xy", scale_units="xy", scale=1, color="0.55",
                    alpha=0.4, width=0.009, label="Fixed pump")
        axes.plot(probes[:, 0], probes[:, 1], color=pump_colour, linestyle="--", zorder=3, label="Probe")
        axes.plot(response[:, 0], response[:, 1], color=generated_colour, label="Mixed response")
        screen_axes(axes, limit)
    figure.legend(*grid[0, 0].get_legend_handles_labels(), loc="outside lower center",
                  ncol=3, frameon=False)
    return figure


@charts.style()
def draw_growth(depths: np.ndarray, amplitude: core.Polarization) -> plt.Figure:
    """The amplitude's vertical path in the phase plane and the power along each crystal."""
    components = transverse_polarization(amplitude)                          # [cases, slices, quadratures, 2]
    power = (components ** 2).sum(axis=(-2, -1))                            # [cases, slices]
    amplitude = components[..., 1]                                          # [cases, slices, quadratures]
    figure, axes = charts.subplots(*growth_size, ncols=2, layout="constrained")
    quadrature_axes, power_axes = axes
    for label, path, intensity, colour in zip(cases, amplitude, power, case_colours):
        quadrature_axes.plot(path[:, 0], path[:, 1], color=colour, label=label)
        quadrature_axes.plot(path[-1, 0], path[-1, 1], "o", color=colour)
        power_axes.plot(depths, intensity, color=colour, label=label)
    # The plane about the paths and the origin they start from, with a margin.
    lowest = np.minimum(amplitude.min(axis=(0, 1)), 0)                       # [2]
    highest = np.maximum(amplitude.max(axis=(0, 1)), 0)
    margin = 0.1 * (highest - lowest).max()
    cross(quadrature_axes, np.stack([lowest - margin, highest + margin], axis=-1), "in phase", "in quadrature")
    power_axes.set(xlabel="depth in the crystal", ylabel="generated power", xlim=(0, 1), xticks=[], yticks=[])
    power_axes.spines[["top", "right"]].set_visible(False)
    # The cases named under both panels, whose colours they share.
    figure.legend(*power_axes.get_legend_handles_labels(), loc="outside lower center", ncol=len(cases),
                  frameon=False)
    return figure


def animate(pumps: core.Field, frames: Iterable[tuple[core.Direction, core.Polarization]]) -> Iterator[np.ndarray]:
    """The response while the crystal turns about the beam, one frame per turn."""
    # The charts' style held while the frames are drawn, for the artists that drawing makes.
    with charts.style():
        view = ResponseView(pumps)
        for bonds, harmonic in frames:
            yield capture(view.update(bonds, harmonic))
        plt.close(view.figure)
