"""Polarization response, mixed fields, and coherent growth drawn from geometric states."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from numga import Extensor, NumpyContext, concatenate, stack
from numga.algebras import VGA2D

from examples.util import charts
from examples.util.animation import capture
from examples.electromagnetism.second_harmonic import core
from numga.topology import Topology
from numga.render import Camera, drawing, Scene, simplices, solids
from examples.util.panels import fitted, labels, offset_panel, overlay, show

# Meeting a spatial direction's worldsheet with the observer's slice gives its spatial point.
space_slice = core.mv.t ^ (core.Direction & core.time)
# Two-dimensional Euclidean plane holding the in-phase and quadrature wave amplitudes as antivectors.
phase_plane = NumpyContext(VGA2D).multivector

pump_colour = "#3275a8"
generated_colour = "#d16a30"
case_colours = ("#3275a8", "#d16a30", "#7b62a3")
# Labels for the three phase-matching crystal profiles.
cases = ("matched", "mismatched", "mismatched, flipped every half turn of slip")
bond_colour, centre_colour, axis_colour = np.full(3, 0.45), np.full(3, 0.3), np.array([0.68, 0.70, 0.75])
label_colour = np.array([0.4, 0.42, 0.47])
# Faint crosshair colour through the origin of each transverse or phase plane.
cross_colour = np.full(3, 0.85)
# Figure pixel dimensions for the response, waveform, growth, and mixing views.
response_size, waveform_size, growth_size = (800, 280), (460, 220), (640, 300)
mixing_panel_size = (260, 290)
# Elevation, azimuth, and half-width of the 3D crystal view around the bond centre.
elevation, azimuth = np.radians(23.0), np.radians(34.0)
half_width = 1.35
# Pixel stroke widths, ball radii, and axis lengths in bond-length units.
bond_width, arrow_width, axis_tip, beam_chevron, end_radius, centre_radius = 3.2, 1.9, 2.6, 7.6, 4.5, 3.2
axis_length, label_distance = 1.15, 1.27
# Beam direction along x + y pointing toward the viewer, perpendicular to y - x (right) and z (up).
beam_step = core.mv.x.dual() + core.mv.y.dual()                              # Direction-
beam = beam_step / beam_step.antinorm()                                      # Direction-
beam_reach = 1.25
# Stroke widths and arrow chevron lengths in pixels for transverse response and mixing panels.
curve_width = charts.LINE_PX
arrow_shaft, arrow_chevron = 3.0, 12.0
pale_shaft, pale_chevron, pale_opacity = 2.3, 9.2, 0.4
# Half-extent of the transverse response screen in pump-amplitude units.
response_limit = 1.15
probe_dashes = (5.5, 2.4)


# --- plumbing -------------------------------------------------------------------------
def transverse(fields: core.Field) -> np.ndarray:
    """Pairs `fields` with the horizontal and vertical polarizations to read out `[..., 2]` transverse
    screen components."""
    polarizations = stack((core.horizontal_polarization, core.vertical_polarization))
    return (fields[..., None] & polarizations).to_array()                     # [..., 2]


def transverse_polarization(polarizations: core.Polarization) -> np.ndarray:
    """Pairs the horizontal and vertical fields with `polarizations` to read out `[..., 2]` transverse
    screen components."""
    fields = stack((core.horizontal_field, core.vertical_field))
    return (fields & polarizations[..., None]).to_array()                     # [..., 2]


def named(axes: plt.Axes, extents: np.ndarray, across: str, up: str) -> None:
    """Frames `axes` to `extents` (`[2, 2]`), hides ticks, and labels the positive crosshair ends with
    `across` and `up`."""
    axes.text(extents[0, 1], 0, f" {across}", ha="left", va="center", color=label_colour)
    axes.text(0, extents[1, 1], up, ha="center", va="bottom", color=label_colour)
    axes.set(xlim=extents[0], ylim=extents[1], aspect="equal")
    axes.set_axis_off()


def screen_axes(axes: plt.Axes, limit: float) -> None:
    """Frames the transverse beam plane to `[-limit, limit]` along `y - x` and `z`."""
    named(axes, np.array([[-limit, limit], [-limit, limit]]), "y - x", "z")


def screens(axes: plt.Axes) -> tuple[Camera, Camera]:
    """Creates native pairing cameras on `axes` for transverse fields (`Field+`) and transverse
    polarizations (`Polarization-`)."""
    return (offset_panel(axes, core.horizontal_polarization, core.vertical_polarization),
            offset_panel(axes, core.horizontal_field, core.vertical_field))


def cross(camera: Camera, limit: float) -> np.ndarray:
    """Renders a faint crosshair of half-length `limit` through the origin of `camera`."""
    along, rise = (plane.dual() / (plane & plane.dual()) for plane in (camera.across, camera.up))
    ends = limit * stack((-along, along, -rise, rise))                       # [4]
    return simplices.strokes(ends, Topology.chains([2, 2]), cross_colour, charts.FRAME_PX, camera,
                             cap=simplices.Cap.BUTT)


def curve(values: Extensor, colour: str, camera: Camera, **style) -> np.ndarray:
    """Renders a polyline stroke through `values` (`[...]`) on `camera`."""
    return simplices.strokes(values, Topology.chains([len(values)]), np.array(to_rgb(colour)), curve_width,
                             camera, **style)


def arrow(value: Extensor, colour, width: float, chevron: float, camera: Camera, opacity: float = 1.0) -> np.ndarray:
    """Renders an arrow stroke from the origin to `value` on `camera`."""
    return simplices.strokes(stack((0 * value, value)), Topology.chains([2]), np.array(to_rgb(colour)), width,
                             camera, chevrons=np.ones(1), chevron=chevron, opacity=opacity)


def legend_line(colour, label: str, **style) -> Line2D:
    """Creates a matplotlib `Line2D` legend proxy matching a GPU-rendered stroke."""
    return Line2D([], [], color=colour, label=label, **style)


def space_scene(view: solids.View) -> Scene:
    """Assembles the static 3D coordinate axes and incoming beam arrow inside `view`."""
    tips = solids.origin + solids.context.include_from_algebra(space_slice(axis_length * core.axes))   # [3] Point-
    reach = solids.context.include_from_algebra(space_slice(beam_reach * beam))                        # Point-
    shaft_radius, tip_radius = arrow_width / 2 * view.pixel, axis_tip * view.pixel
    beam_colour = np.array(to_rgb(pump_colour))
    parts = [(solids.rods(solids.origin, tips, shaft_radius), axis_colour),
             (solids.ball(tips, tip_radius)[:, None], axis_colour),
             (solids.ball(solids.origin, centre_radius * view.pixel)[None, None], centre_colour)]
    beam_arrow = simplices.tubes(stack((solids.origin - reach, solids.origin + reach)), Topology.chains([2]),
                                 beam_colour, arrow_width, view, gloss=solids.gloss, chevrons=np.ones(1),
                                 chevron=beam_chevron)
    return Scene(view, np.ones(3), parts, [(beam_arrow, 1.0)], inks=[], translucent=[])


def crystal_image(bonds: core.Direction, scene: Scene) -> np.ndarray:
    """Draws the four tetrahedral bonds (`[4] Direction-`) as rods and tip spheres inside `scene`."""
    ends = solids.origin + solids.context.include_from_algebra(space_slice(bonds))              # [4] Point-
    view = scene.view
    return scene.draw([(solids.rods(solids.origin, ends, bond_width / 2 * view.pixel), bond_colour),
                       (solids.ball(ends, end_radius * view.pixel)[:, None], np.array(to_rgb(generated_colour)))],
                       inks=[], translucent=[], meshes=[])


class ResponseView:
    """Stateful three-panel figure updating the 3D crystal, transverse polarization curve, and polar
    intensity lobe across crystal rotations."""

    def __init__(self, pumps: core.Field) -> None:
        # Polar heading angles of the pump fields across the beam screen.
        self.angles = np.arctan2((pumps & core.vertical_polarization).to_array(),
                                 (pumps & core.horizontal_polarization).to_array())             # [...]
        self.selected_angle = self.angles[0]

        width, height = response_size
        self.figure = plt.figure(figsize=(width / charts.DPI, height / charts.DPI), dpi=charts.DPI,
                                 layout="constrained")
        crystal_axes = self.figure.add_subplot(1, 3, 1)
        response_axes = self.figure.add_subplot(1, 3, 2)
        self.intensity_axes = self.figure.add_subplot(1, 3, 3, projection="polar")

        screen_axes(response_axes, response_limit)
        response_axes.legend(handles=[legend_line(pump_colour, "Pump"), legend_line(generated_colour, "Second harmonic")],
                             loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=2, frameon=False)

        self.intensity_curve, = self.intensity_axes.plot([], [], color=generated_colour)
        self.intensity_fill, = self.intensity_axes.fill([], [], color=generated_colour, alpha=0.08)
        self.selected_power, = self.intensity_axes.plot([], [], "o", color=pump_colour)
        self.intensity_axes.yaxis.set_major_locator(MaxNLocator(4))
        self.intensity_axes.set_yticklabels([])
        self.intensity_axes.set_thetagrids([0, 45, 90, 135, 180, 225, 270, 315])
        self.intensity_axes.set_rlabel_position(22.5)

        crystal_axes.set_axis_off()
        self.figure.get_layout_engine().execute(self.figure)
        self.figure.set_layout_engine("none")
        fields, self.polarizations = screens(response_axes)
        self.pumps = [cross(fields, response_limit), curve(pumps, pump_colour, fields, opacity=0.3),
                      arrow(pumps[0], pump_colour, arrow_shaft, arrow_chevron, fields)]
        self.response = overlay(response_axes, self.pumps)
        self.view = fitted(solids.View(solids.aim(solids.origin, elevation, azimuth), np.full(2, half_width), 1),
                           crystal_axes)
        self.crystal = show(crystal_axes, np.full((*self.view.shape, 3), 255, dtype=np.uint8), self.view)
        self.scene = space_scene(self.view)
        label_reaches = concatenate((label_distance * core.axes, beam_reach * 1.12 * beam[None]))       # [4] Direction-
        label_places = solids.origin + solids.context.include_from_algebra(space_slice(label_reaches))  # [4] Point-
        labels(crystal_axes, self.view, label_places, ["x", "y", "z", "beam"], ha="center", va="center",
               color=label_colour)

    def update(self, bonds: core.Direction, harmonic: core.Polarization) -> plt.Figure:
        intensity = harmonic.norm_squared().to_array()                                # [...]
        self.crystal.set_data(crystal_image(bonds, self.scene))
        self.response.set_data(drawing.composited(self.pumps + [
            curve(harmonic, generated_colour, self.polarizations, opacity=0.65),
            arrow(harmonic[0], generated_colour, arrow_shaft, arrow_chevron, self.polarizations)]))
        self.intensity_curve.set_data(self.angles, intensity)
        self.intensity_fill.set_xy(np.stack((self.angles, intensity), axis=-1))
        self.selected_power.set_data([self.selected_angle], [intensity[0]])
        self.intensity_axes.set_ylim(0, 1.12 * float(np.max(intensity)))
        return self.figure


@charts.style()
def draw_response(pumps: core.Field, bonds: core.Direction, harmonic: core.Polarization) -> plt.Figure:
    """Draws a single three-panel crystal orientation, transverse polarization curve, and polar power
    plot."""
    return ResponseView(pumps).update(bonds, harmonic)


@charts.style()
def draw_waveform(phase: np.ndarray, pump: core.Field, harmonic: core.Polarization) -> plt.Figure:
    """Plots the horizontal pump waveform and vertical doubled-frequency polarization over two pump
    cycles."""
    cycles = phase / (2 * np.pi)                                               # [...]
    pump_field = transverse(pump)[..., 0]                                      # [...]
    harmonic_field = transverse_polarization(harmonic)[..., 1]                 # [...]
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
    """Draws the probe circle and its linear mixed response alongside each fixed pump at a shared
    display scale."""
    probe_readings = probes[..., None] & stack((core.horizontal_polarization, core.vertical_polarization))   # [..., 2] Scalar+
    response_readings = stack((core.horizontal_field, core.vertical_field)) & generated[..., None]           # [2, ..., 2] Scalar+
    limit = float((1.25 * stack((probe_readings.abs().max(), response_readings.abs().max())).max()).to_array())
    # Pale arrows indicate the fixed pump directions at a display-scaled length.
    pump_arrows = 0.85 * limit * pumps                                       # [2] Field+
    figure, grid = charts.subplots(len(pump_arrows) * mixing_panel_size[0], mixing_panel_size[1],
                                   ncols=len(pump_arrows), squeeze=False, layout="constrained")
    for axes in grid[0]:
        screen_axes(axes, limit)
    figure.legend(handles=[legend_line("0.55", "Fixed pump", alpha=0.4),
                           legend_line(pump_colour, "Probe", linestyle="--"),
                           legend_line(generated_colour, "Mixed response")],
                  loc="outside lower center", ncol=3, frameon=False)
    for axes, pump, response in zip(grid[0], pump_arrows, generated):
        fields, polarizations = screens(axes)
        overlay(axes, [cross(fields, limit),
                       arrow(pump, "0.55", pale_shaft, pale_chevron, fields, pale_opacity),
                       curve(response, generated_colour, polarizations),
                       curve(probes, pump_colour, fields, dashes=probe_dashes, cap=simplices.Cap.BUTT)])
    return figure


@charts.style()
def draw_growth(depths: np.ndarray, amplitude: core.Polarization) -> plt.Figure:
    """Plots the vertical quadrature trajectories in the phase plane beside the accumulated power
    along the three crystals."""
    power = amplitude.norm_squared().sum(axis=-1).to_array()                 # [3, ...]
    vertical = core.vertical_field & amplitude                               # [3, ..., 2] Scalar+
    figure, axes = charts.subplots(*growth_size, ncols=2, layout="constrained")
    quadrature_axes, power_axes = axes
    for label, intensity, colour in zip(cases, power, case_colours):
        power_axes.plot(depths, intensity, color=colour, label=label)
    lowest = vertical.min(axis=(0, 1)).clip(-np.inf, 0)                      # [2] Scalar+
    highest = vertical.max(axis=(0, 1)).clip(0, np.inf)                      # [2] Scalar+
    margin = 0.1 * (highest - lowest).max()
    extents = stack((lowest - margin, highest + margin), axis=-1).to_array()  # [2, 2]
    named(quadrature_axes, extents, "in phase", "in quadrature")
    components = vertical.to_array()                                         # [3, ..., 2]
    paths = phase_plane.x.dual() * components[..., 0] + phase_plane.y.dual() * components[..., 1]   # [3, ...] Antivector-
    ends = stack((phase_plane.x.dual() * extents[0], phase_plane.y.dual() * extents[1])).reshape(-1)  # [4] Antivector-
    camera = offset_panel(quadrature_axes, phase_plane.x, phase_plane.y)
    traced = [simplices.strokes(path, Topology.chains([len(path)]), np.array(to_rgb(colour)), curve_width,
                                camera) for path, colour in zip(paths, case_colours)]
    reached = simplices.markers(paths[:, -1], np.array([to_rgb(colour) for colour in case_colours]),
                                charts.MARKER_PX / 2, camera)
    overlay(quadrature_axes, [simplices.strokes(ends, Topology.chains([2, 2]), cross_colour,
                                                charts.FRAME_PX, camera, cap=simplices.Cap.BUTT),
                              *traced, reached])
    power_axes.set(xlabel="depth in the crystal", ylabel="generated power", xlim=(0, 1), xticks=[], yticks=[])
    power_axes.spines[["top", "right"]].set_visible(False)
    figure.legend(*power_axes.get_legend_handles_labels(), loc="outside lower center", ncol=len(cases),
                  frameon=False)
    return figure


def animate(pumps: core.Field, frames: Iterable[tuple[core.Direction, core.Polarization]]) -> Iterator[np.ndarray]:
    """Yields captured RGB frames as the crystal turns about the beam."""
    with charts.style():
        view = ResponseView(pumps)
        for bonds, harmonic in frames:
            yield capture(view.update(bonds, harmonic))
        plt.close(view.figure)
