"""Polarization response, mixed fields and coherent growth drawn from geometric states."""

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
from examples.util.wgsl import fitted, labels, show, simplices, solids
from examples.util.wgsl.camera import Camera, flat_panel, panel
from examples.util.wgsl.scene import Scene

# The observer sees the point where its time line meets the spatial slice.
space_slice = core.mv.t ^ (core.Direction & core.time)
# The phase plane of a wave's amplitude, its two quadratures in phase across and in quadrature up, its points
# antivectors.
phase_plane = NumpyContext(VGA2D).multivector

pump_colour = "#3275a8"
generated_colour = "#d16a30"
case_colours = ("#3275a8", "#d16a30", "#7b62a3")
# The crystals of the phase-matching scene, in its order.
cases = ("matched", "mismatched", "mismatched, flipped every half turn of slip")
bond_colour, centre_colour, axis_colour = np.full(3, 0.45), np.full(3, 0.3), np.array([0.68, 0.70, 0.75])
label_colour = np.array([0.4, 0.42, 0.47])
# The faint cross through the origin of a plane.
cross_colour = np.full(3, 0.85)
# The charts' sizes in pixels: the response, the crystal beside the response and its power, all of the column; the
# waveform, two curves over two periods; the mixing, a panel to a pump above the legend; the phase matching, the
# phase plane beside the power along the crystals.
response_size, waveform_size, growth_size = (800, 280), (460, 220), (640, 300)
mixing_panel_size = (260, 290)
# The crystal's view: elevation and azimuth in radians, azimuth zero looking along +y, and the half width of the
# square view about the bonds' centre.
elevation, azimuth = np.radians(23.0), np.radians(34.0)
half_width = 1.35
# The widths of the bonds and the axes' arrows, the radius of the ball at an axis's tip, the length of the strokes of
# the chevron at the beam's head, and the radii of the bonds' ends and their centre, in pixels; how far the axes reach
# and their names lie, in bond lengths.
bond_width, arrow_width, axis_tip, beam_chevron, end_radius, centre_radius = 3.2, 1.9, 2.6, 7.6, 4.5, 3.2
axis_length, label_distance = 1.15, 1.27
# The beam along the crystal's face diagonal, x + y, coming towards the screen's viewer, across which the screen's
# directions are y - x, to the right, and z, up; how far the beam reaches either side of the crystal, in bond lengths.
beam_step = core.mv.x.dual() + core.mv.y.dual()                              # [] Direction-
beam = beam_step / beam_step.squared().square_root()                         # [] Direction-
beam_reach = 1.25
# The response's and the mixing's widths, and the lengths of the strokes of the chevrons at the arrows' tips, in
# pixels: curves, the pump and second harmonic's arrows, the fixed pumps' pale arrows, and the probe's dashes.
curve_width = charts.LINE_PX
arrow_shaft, arrow_chevron = 3.0, 12.0
pale_shaft, pale_chevron, pale_opacity = 2.3, 9.2, 0.4
# The extent of the response's plane across the beam, either way, in units of the pump.
response_limit = 1.15
probe_dashes = (5.5, 2.4)


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


def named(axes: plt.Axes, extents: np.ndarray, across: str, up: str) -> None:
    """A plane within extents `[2, 2]`, lowest and highest across and up, without scales: the lines of its cross through
    the origin named just past their ends, clear of what the plane holds."""
    axes.text(extents[0, 1], 0, f" {across}", ha="left", va="center", color=label_colour)
    axes.text(0, extents[1, 1], up, ha="center", va="bottom", color=label_colour)
    axes.set(xlim=extents[0], ylim=extents[1], aspect="equal")
    axes.set_axis_off()


def screen_axes(axes: plt.Axes, limit: float) -> None:
    """The plane across the beam, one extent on both directions, named by the directions of space along it."""
    named(axes, np.array([[-limit, limit], [-limit, limit]]), "$y - x$", "$z$")


def screens(axes: plt.Axes) -> tuple[Camera, Camera]:
    """The cameras of the screen across the beam, y - x to the right and z up, each reading values as they are: fields
    by the polarizations along those directions, `horizontal_polarization & field`, and polarizations by the fields
    along them, `horizontal_field & polarization`."""
    return (panel(axes, core.horizontal_polarization, core.vertical_polarization),
            panel(axes, core.horizontal_field, core.vertical_field))


def cross(camera: Camera, limit: float) -> np.ndarray:
    """A faint cross through the origin of the plane across the beam, `limit` either way along the elements the
    camera reads as one across and one up, the duals of its planes over their own readings: its picture."""
    along, rise = (plane.dual() / (plane & plane.dual()) for plane in (camera.across, camera.up))
    ends = limit * stack((-along, along, -rise, rise))                       # [4]
    return simplices.strokes(ends, simplices.Mesh.chains([2, 2]), cross_colour, charts.FRAME_PX, camera,
                             cap=simplices.Cap.BUTT)


def curve(values: Extensor, colour: str, camera: Camera, **style) -> np.ndarray:
    """A curve through fields or polarizations across the beam `[samples]`, in order, as a stroke."""
    return simplices.strokes(values, simplices.Mesh.chains([len(values)]), np.array(to_rgb(colour)), curve_width,
                             camera, **style)


def arrow(value: Extensor, colour, width: float, chevron: float, camera: Camera, opacity: float = 1.0) -> np.ndarray:
    """An arrow from the origin to a field or a polarization across the beam `[]`: a stroke with a chevron of strokes
    `chevron` pixels long at its tip, covering as much as `opacity`."""
    return simplices.strokes(stack((0 * value, value)), simplices.Mesh.chains([2]), np.array(to_rgb(colour)), width,
                             camera, chevrons=np.ones(1), chevron=chevron, opacity=opacity)


def legend_line(colour, label: str, **style) -> Line2D:
    """A legend's entry for a stroke the GPU draws: a line in its colour and style."""
    return Line2D([], [], color=colour, label=label, **style)


def space_scene(view: solids.View) -> Scene:
    """The frame of space, its centre and the beam through it, for the bonds to be drawn among: each axis an arrow, a
    rod with a ball at its tip, and the beam a tube with a chevron at its head."""
    # The axes' tips, embedded about the drawing's origin, and the beam's reach, a direction of space, embedded in the
    # drawing's projective space as a point at infinity.
    tips = solids.origin + solids.context.include_from_algebra(space_slice(axis_length * core.axes))   # [axes] Point-
    reach = solids.context.include_from_algebra(space_slice(beam_reach * beam))                 # [] Point-
    shaft_radius, tip_radius = arrow_width / 2 * view.pixel, axis_tip * view.pixel
    beam_colour = np.array(to_rgb(pump_colour))
    parts = [(solids.rods(solids.origin, tips, shaft_radius), axis_colour),
             (solids.ball(tips, tip_radius)[:, None], axis_colour),
             (solids.ball(solids.origin, centre_radius * view.pixel)[None, None], centre_colour)]
    beam_arrow = simplices.tubes(stack((solids.origin - reach, solids.origin + reach)), simplices.Mesh.chains([2]),
                                 beam_colour, arrow_width, view, gloss=solids.gloss, chevrons=np.ones(1),
                                 chevron=beam_chevron)
    return Scene(view, np.ones(3), parts, [(beam_arrow, 1.0)], inks=[], translucent=[])


def crystal_image(bonds: core.Direction, scene: Scene) -> np.ndarray:
    """The bonds `[bonds] Direction-` as rods from their shared centre to their ends, the tetrahedral geometry,
    among the scene's frame of space and beam."""
    # The bonds' ends, sliced by the observer, embedded about the drawing's origin, the bonds' centre.
    ends = solids.origin + solids.context.include_from_algebra(space_slice(bonds))              # [bonds] Point-
    view = scene.view
    return scene.draw([(solids.rods(solids.origin, ends, bond_width / 2 * view.pixel), bond_colour),
                       (solids.ball(ends, end_radius * view.pixel)[:, None], np.array(to_rgb(generated_colour)))],
                       inks=[], translucent=[], meshes=[])


class ResponseView:
    """One response figure for a fixed set of pumps, its artists updated for each crystal."""

    def __init__(self, pumps: core.Field) -> None:
        # The polar chart's angle of each pump, its heading across the beam read from its pairings with the
        # polarizations along the screen's directions: a turn's logarithm would give it too, but has no plane at the
        # half turn.
        self.angles = np.arctan2((pumps & core.vertical_polarization).to_array(),
                                 (pumps & core.horizontal_polarization).to_array())   # [angles]
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

        # The figure laid out once, the crystal's axes as bare as its image leaves them, and kept so: at every
        # resolution it is drawn at, the panels stay put and the crystal's image fills its room.
        crystal_axes.set_axis_off()
        self.figure.get_layout_engine().execute(self.figure)
        self.figure.set_layout_engine("none")
        # The pumps across the beam, the first as an arrow, under the second harmonic each crystal drives.
        fields, self.polarizations = screens(response_axes)
        self.pumps = [cross(fields, response_limit), curve(pumps, pump_colour, fields, opacity=0.3),
                      arrow(pumps[0], pump_colour, arrow_shaft, arrow_chevron, fields)]
        self.response = simplices.overlay(response_axes, self.pumps)
        # The crystal's image, blank until the bonds are drawn, as many pixels across as its room holds, with each
        # axis named just past its arrow's tip.
        self.view = fitted(solids.View(solids.aim(solids.origin, elevation, azimuth), np.full(2, half_width), 1),
                           crystal_axes)
        self.crystal = show(crystal_axes, np.full((*self.view.shape, 3), 255, dtype=np.uint8), self.view)
        self.scene = space_scene(self.view)
        # Each axis named just past its arrow's tip, and the beam past its head.
        label_reaches = concatenate((label_distance * core.axes, beam_reach * 1.12 * beam[None]))   # [labels] Direction-
        label_places = (solids.origin + solids.context.include_from_algebra(space_slice(label_reaches)))                             # [labels] Point-
        labels(crystal_axes, self.view, label_places, ["x", "y", "z", "beam"], ha="center", va="center",
               color=label_colour)

    def update(self, bonds: core.Direction, harmonic: core.Polarization) -> plt.Figure:
        # The doubled light's power, the squared size of its polarization across the beam.
        intensity = harmonic.norm_squared().to_array()                       # [angles]
        self.crystal.set_data(crystal_image(bonds, self.scene))
        self.response.set_data(simplices.composited(self.pumps + [
            curve(harmonic, generated_colour, self.polarizations, opacity=0.65),
            arrow(harmonic[0], generated_colour, arrow_shaft, arrow_chevron, self.polarizations)]))
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
    # The plane as wide as the largest reading across the beam of the probes, by the polarizations along the screen's
    # directions, and of the responses, by the fields along them.
    probe_readings = probes[..., None] & stack((core.horizontal_polarization, core.vertical_polarization))   # [angles, 2] Scalar+
    response_readings = stack((core.horizontal_field, core.vertical_field)) & generated[..., None]   # [cases, angles, 2] Scalar+
    limit = float((1.25 * stack((probe_readings.abs().max(), response_readings.abs().max())).max()).to_array())
    # The pale arrows show the fixed pumps' directions at a display-only length.
    pump_arrows = 0.85 * limit * pumps                                       # [cases] Field+
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
        simplices.overlay(axes, [cross(fields, limit),
                                 arrow(pump, "0.55", pale_shaft, pale_chevron, fields, pale_opacity),
                                 curve(response, generated_colour, polarizations),
                                 curve(probes, pump_colour, fields, dashes=probe_dashes, cap=simplices.Cap.BUTT)])
    return figure


@charts.style()
def draw_growth(depths: np.ndarray, amplitude: core.Polarization) -> plt.Figure:
    """The amplitude's vertical path in the phase plane and the power along each crystal."""
    # The power, the squared sizes of both quadratures' polarizations across the beam; and the vertical component
    # of each, which the paths in the phase plane trace.
    power = amplitude.norm_squared().sum(axis=-1).to_array()                 # [cases, slices]
    vertical = core.vertical_field & amplitude                                # [cases, slices, quadratures] Scalar+
    figure, axes = charts.subplots(*growth_size, ncols=2, layout="constrained")
    quadrature_axes, power_axes = axes
    for label, intensity, colour in zip(cases, power, case_colours):
        power_axes.plot(depths, intensity, color=colour, label=label)
    # The plane about the paths and the origin they start from, with a margin.
    lowest = vertical.min(axis=(0, 1)).clip(-np.inf, 0)                      # [quadratures] Scalar+
    highest = vertical.max(axis=(0, 1)).clip(0, np.inf)                      # [quadratures] Scalar+
    margin = 0.1 * (highest - lowest).max()
    extents = stack((lowest - margin, highest + margin), axis=-1).to_array()  # [2, 2]
    named(quadrature_axes, extents, "in phase", "in quadrature")
    # Each case's path through the phase plane, ending in a dot, over a faint cross through the origin.
    components = vertical.to_array()                                          # [cases, slices, quadratures]
    paths = phase_plane.x.dual() * components[..., 0] + phase_plane.y.dual() * components[..., 1]   # [cases, slices] Antivector-
    ends = stack((phase_plane.x.dual() * extents[0], phase_plane.y.dual() * extents[1])).reshape(-1)  # [4] Antivector-
    camera = flat_panel(quadrature_axes, VGA2D)
    traced = [simplices.strokes(path, simplices.Mesh.chains([len(path)]), np.array(to_rgb(colour)), curve_width,
                                camera) for path, colour in zip(paths, case_colours)]
    reached = simplices.markers(paths[:, -1], np.array([to_rgb(colour) for colour in case_colours]),
                                charts.MARKER_PX / 2, camera)
    simplices.overlay(quadrature_axes, [simplices.strokes(ends, simplices.Mesh.chains([2, 2]), cross_colour,
                                                          charts.FRAME_PX, camera, cap=simplices.Cap.BUTT),
                                        *traced, reached])
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
