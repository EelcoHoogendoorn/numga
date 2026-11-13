"""Drawing for a p-n junction, a MOSFET and a solar cell. The junction: its band diagram, the conduction and valence
band edges and the electrons' and holes' quasi-Fermi levels along the bar, energy up and the p side on the left, the
bands tinted; and beside it, as the bias sweeps, the current against the voltage, traced as it goes. The MOSFET: its
cross-section, the silicon tinted by its electrons, blue, and its holes, red, its surface beneath the oxide drawn as a
band of its own tint, the oxide grey under the dark gate, the contacts' metal dark; and beside it the drain's current
against its voltage, an output curve traced for each gate voltage as the drain sweeps. The solar cell: a window of its
cross-section about the middle of a gap between two fingers, tiled by its period, the silicon tinted warm by the
light's generation and white in the fingers' shadow, its top layer and its back layer bands of their own tints, the
fingers dark on the surface, and the holes' flow streaked over it; beside it, as the light comes on and the bias rises,
the current it delivers against its voltage, and as its fingers spread, its power against their spacing; and its power
against the spacing as a chart.

The bar's vertices are read by the plane x, the cross-sections' by the planes x and y over their weight, which the plane
w reads; the energies, densities, currents, voltages and powers come as scalars, read off the states by the example's
own functions, and the holes' flow as a direction in each simplex. The band diagram and the curves are charts; the
cross-sections are drawn on the GPU in their own algebra.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure

from numga import concatenate, stack
from numga.extensor import Extensor
from numga.render import contours, drawing, flow, simplices
from numga.render.camera import Camera
from numga.topology import Topology
from examples.util import charts
from examples.util.animation import capture
from examples.util.panels import panel, pinned

# The band edges' colour, the electrons' and the holes', the faint junction's, and the curves traced in the dark and
# under light; the bands' tints where they hold carriers at the doping's density, fading to white at the intrinsic one.
edge_colour, electron_colour, hole_colour = "#2b2f33", "#2166ac", "#b2182b"
conduction_tint, valence_tint = np.array(to_rgb("#c4d6ec")), np.array(to_rgb("#efcdcd"))
junction_colour = "#c9ced3"
dark_colour, lit_colour = "#2b2f33", "#e08214"
# The quasi-Fermi levels' dashes, a dash and a gap as long, in points, the holes' between the electrons', so that where
# the two levels meet they alternate; and the band edges' width, half again a chart's line.
dash = 3.0
electron_dashes, hole_dashes = (0.0, (dash, dash)), (dash, (dash, dash))
edge_width = 1.5 * charts.LINE_WIDTH
# The panels' sizes in pixels: the band diagram, and the current against the voltage beside it.
band_width_px, band_height_px = 340, 240
curve_width_px = 240
# The room the labels at the band diagram's ends take, as a share of the bar's length, and the voltages' ticks, in
# volts apart; the room the current's chart leaves past the sweep's ends, as a share of its range, for the dot there and
# the voltage's label.
label_room, tick_volts, sweep_room = 0.16, 0.2, 0.08


def band_panel(axis: plt.Axes, places: np.ndarray, window: tuple[float, float], intrinsic: float
               ) -> Callable[[np.ndarray, np.ndarray], None]:
    """The band diagram's axes over the bar's places `[vertices]` and the energies in the window, and a function that
    draws the conduction band's edge, the valence band's and the two quasi-Fermi levels `[4, vertices]` on it, each
    band tinted along the bar by the carriers it holds `[2, vertices]`, on a logarithmic scale from this intrinsic
    density to the doping's, one."""
    left, right = places[0], places[-1]
    axis.set(xlim=(left - label_room * (right - left), right + label_room * (right - left)), ylim=window, xticks=[],
             yticks=[])
    axis.spines[["top", "right", "bottom", "left"]].set_visible(False)
    axis.plot([0.0, 0.0], window, color=junction_colour, linewidth=charts.FRAME_WIDTH, zorder=0)
    for side, place in (("p", left / 2), ("n", right / 2)):
        axis.text(place, window[1], side, ha="center", va="top", color=edge_colour)
    # Each band a strip of quadrilaterals, one over each edge, from its edge to the window's end.
    bands = [PolyCollection([], antialiased=False, linewidth=0, zorder=-1) for _ in range(2)]
    for band in bands:
        axis.add_collection(band)
    conduction, = axis.plot([], [], color=edge_colour, linewidth=edge_width)
    valence, = axis.plot([], [], color=edge_colour, linewidth=edge_width)
    electrons, = axis.plot([], [], color=electron_colour, linestyle=electron_dashes)
    holes, = axis.plot([], [], color=hole_colour, linestyle=hole_dashes)
    labels = [axis.text(0, 0, name, color=colour, va="center", ha=align)
              for name, colour, align in ((r"$E_c$", edge_colour, "left"), (r"$E_v$", edge_colour, "left"),
                                          (r"$E_{Fn}$", electron_colour, "left"), (r"$E_{Fp}$", hole_colour, "right"))]

    def draw(energies: np.ndarray, densities: np.ndarray) -> None:
        for band, edge, end, density, tint in zip(bands, energies[:2], window[::-1], densities,
                                                  (conduction_tint, valence_tint)):
            # Each edge's share of the band's full tint: its mean density's logarithm, from the intrinsic density's
            # to the doping's.
            shares = np.clip(np.log((density[:-1] + density[1:]) / 2 / intrinsic) / np.log(1 / intrinsic), 0, 1)
            band.set_verts(np.stack([np.stack([places[:-1], edge[:-1]], -1), np.stack([places[1:], edge[1:]], -1),
                                     np.stack([places[1:], np.full(len(places) - 1, end)], -1),
                                     np.stack([places[:-1], np.full(len(places) - 1, end)], -1)], axis=1))
            band.set_facecolor(1 - shares[:, None] * (1 - tint))
        for line, energy in zip((conduction, valence, electrons, holes), energies):
            line.set_data(places, energy)
        # The band edges and the electrons' level named at the n end, the holes' at the p end, each beside its line.
        gap = 0.02 * (right - left)
        for label, energy in zip(labels[:3], energies[:3]):
            label.set_position((right + gap, energy[-1]))
        labels[3].set_position((left - gap, energies[3][0]))

    return draw


def draw_bands(vertices: Extensor, energies: Extensor, densities: Extensor, intrinsic: float,
               window: tuple[float, float]) -> plt.Figure:
    """The band diagram of the bar at its vertices `Point-[vertices]`, its four energies `[4] Scalar+[vertices]` and
    its carriers' densities `[2] Scalar+[vertices]` tinting the bands from this intrinsic density up, in this window
    of energy."""
    mv = vertices.context.multivector
    places = np.asarray((mv.x & vertices.batch()).to_array())
    with charts.style():
        figure, axis = charts.subplots(band_width_px, band_height_px)
        figure.subplots_adjust(left=0.0, right=1.0, bottom=0.02, top=0.98)
        band_panel(axis, places, window, intrinsic)(np.asarray(energies.batch().to_array()).reshape(4, -1),
                                                    np.asarray(densities.batch().to_array()).reshape(2, -1))
    return figure


def animate_forming(vertices: Extensor, bands_of: Callable[[Extensor], Extensor],
                    carriers_of: Callable[[Extensor], Extensor], intrinsic: float, window: tuple[float, float],
                    states: Iterable[Extensor]) -> Iterator[np.ndarray]:
    """The band diagram of the bar at its vertices `Point-[vertices]` at each state as it comes, its four energies
    `[4] Scalar+[vertices]` given by `bands_of` and its carriers' densities `[2] Scalar+[vertices]` by `carriers_of`,
    tinting the bands from this intrinsic density up, in this window of energy."""
    mv = vertices.context.multivector
    places = np.asarray((mv.x & vertices.batch()).to_array())
    with charts.style():
        figure, axis = charts.subplots(band_width_px, band_height_px)
        figure.subplots_adjust(left=0.0, right=1.0, bottom=0.02, top=0.98)
        draw = band_panel(axis, places, window, intrinsic)
        for state in states:
            draw(np.asarray(bands_of(state).batch().to_array()).reshape(4, -1),
                 np.asarray(carriers_of(state).batch().to_array()).reshape(2, -1))
            yield capture(figure)
        plt.close(figure)


def animate_sweep(vertices: Extensor, bands_of: Callable[[Extensor], Extensor],
                  carriers_of: Callable[[Extensor], Extensor], intrinsic: float,
                  terminal_of: Callable[[Extensor], tuple[Extensor, Extensor]], window: tuple[float, float],
                  voltages: tuple[float, float], currents: tuple[float, float], volt: float, colour: str,
                  beneath: list[list[Extensor]], states: Iterable[Extensor]) -> Iterator[np.ndarray]:
    """The band diagram at each state as it comes, as `animate_forming` draws it, and beside it the current against the
    voltage, both read off the state by `terminal_of`, traced in this colour up to the state's, in these windows of
    voltage and current; the voltage in thermal units, labelled in volts, `volt` thermal units each. Beneath the trace,
    faint, the curves of sweeps before, their states read as these are."""
    mv = vertices.context.multivector
    places = np.asarray((mv.x & vertices.batch()).to_array())
    with charts.style():
        figure, (band_axis, curve_axis) = charts.subplots(
            band_width_px + curve_width_px, band_height_px, ncols=2,
            gridspec_kw=dict(width_ratios=(band_width_px, curve_width_px)))
        figure.subplots_adjust(left=0.0, right=0.97, bottom=0.1, top=0.92, wspace=0.08)
        draw = band_panel(band_axis, places, window, intrinsic)
        # The current against the voltage: axes through nothing, and the voltage's ticks in volts, but at nothing, where
        # the current's axis crosses.
        room = sweep_room * (voltages[1] - voltages[0])
        curve_axis.set(xlim=(voltages[0] - room, voltages[1] + room), ylim=currents, yticks=[])
        ticks = np.arange(np.ceil(voltages[0] / volt / tick_volts), np.floor(voltages[1] / volt / tick_volts) + 1)
        ticks = ticks[ticks != 0] * tick_volts
        curve_axis.set_xticks(ticks * volt, [f"{tick:.1f}" for tick in ticks])
        curve_axis.spines[["top", "right"]].set_visible(False)
        curve_axis.spines["bottom"].set_position(("data", 0.0))
        curve_axis.spines["left"].set_position(("data", 0.0))
        curve_axis.annotate("V", (1, 0), xycoords=("axes fraction", "data"), xytext=(0, 4),
                            textcoords="offset pixels", ha="right", va="bottom")
        curve_axis.annotate("I", (0, 1), xycoords=("data", "axes fraction"), xytext=(5, 0),
                            textcoords="offset pixels", ha="left", va="top")
        for sweep in beneath:
            curve = [[float(reading.to_array()) for reading in terminal_of(state)] for state in sweep]
            curve_axis.plot(*np.transpose(curve), color=dark_colour, alpha=0.35)
        trace, = curve_axis.plot([], [], color=colour)
        dot, = curve_axis.plot([], [], "o", color=colour)
        traced = []
        for state in states:
            draw(np.asarray(bands_of(state).batch().to_array()).reshape(4, -1),
                 np.asarray(carriers_of(state).batch().to_array()).reshape(2, -1))
            voltage, current = terminal_of(state)
            traced.append((float(voltage.to_array()), float(current.to_array())))
            trace.set_data(*np.transpose(traced))
            dot.set_data([traced[-1][0]], [traced[-1][1]])
            yield capture(figure)
        plt.close(figure)


# --- the MOSFET's cross-section --------------------------------------------------------
# The silicon's tints where it holds electrons and holes at their full densities, fading to white where it holds none;
# the oxide's grey and the metal's of the gate and the contacts.
electron_tint, hole_tint = np.array(to_rgb("#3a78b8")), np.array(to_rgb("#dca2a2"))
oxide_colour, metal_colour = np.array(to_rgb("#d4d7da")), np.array(to_rgb("#3b4a57"))
# The silicon's surface beneath the oxide, a band of its own tint, the channel being far thinner than a pixel; the
# gate's and the body contact's strokes; in pixels.
surface_width, gate_width, body_width = 4.0, 4.0, 3.0
# The output curves' colours, one for each gate voltage, light to dark as the gate rises, the dot's, the drain
# voltages' ticks, in volts apart, and the room past the largest drain voltage, a share of it, for the dot there.
curve_colours = ("#9ecae1", "#4292c6", "#08306b")
dot_colour, drain_tick_volts, drain_room = "#2b2f33", 0.5, 0.04
# The panels' sizes in pixels: the cross-section, its margin as a share of its width, and the output curves beside it.
section_width_px, section_height_px, section_margin = 360, 200, 0.02
outputs_width_px = 250


def section_axes(axis: plt.Axes, vertices: Extensor) -> Camera:
    """The cross-section's panel over its vertices `Point-[vertices]`, read by the planes x and y over their weight, the
    plane w's reading, the whole section in view with a thin margin, and the camera that draws into it at its own
    pixels."""
    mv = vertices.context.multivector
    points = vertices.batch()
    across, up = (mv.x & points).to_array(), (mv.y & points).to_array()
    margin = section_margin * (across.max() - across.min())
    axis.set(xlim=(across.min() - margin, across.max() + margin), ylim=(up.min() - margin, up.max() + margin))
    axis.set_aspect("equal")
    axis.set_axis_off()
    return panel(axis, mv.x, mv.y, mv.w)


def tints(electrons: np.ndarray, holes: np.ndarray, full: np.ndarray) -> np.ndarray:
    """The silicon's colour where it holds these electrons and holes `[...]`, each carrier's tint by its share of its
    full density `[carriers]`, `[..., 3]`."""
    electron_share, hole_share = np.clip(electrons / full[0], 0, 1), np.clip(holes / full[1], 0, 1)
    return np.clip(1 - electron_share[..., None] * (1 - electron_tint) - hole_share[..., None] * (1 - hole_tint), 0, 1)


def silicon_layers(topology: Topology, vertices: Extensor, semiconductor: Extensor, electrons: np.ndarray,
                   holes: np.ndarray, full: np.ndarray, camera: Camera) -> list:
    """The silicon tinted by its electrons and holes `[vertices]` each, and its surface beneath the oxide and the
    contacts as a band of the same tint: the silicon's border where it meets the oxide's triangles."""
    mv = vertices.context.multivector
    points = vertices.batch()
    silicon = semiconductor.batch().to_array().reshape(-1) > 0                # [simplices]
    border = topology.subset(silicon, -1).border
    surface = border.subset(np.isin(border.elements[-1], topology.elements[-1][~silicon]).all(axis=-1), -1)
    colours = tints(electrons, holes, full)                                   # [vertices, 3]
    # The band: the surface swept down from the oxide by its width, a copy of the vertices lowered, tinted as the
    # surface is at each vertex and between them in turn.
    band = surface.product(Topology.chains([2]))
    lowered = points - surface_width * camera.pixel * mv.y.dual()
    return [contours.picture(points, topology.subset(silicon, -1), np.zeros(topology.n_vertices), colours, camera),
            contours.picture(concatenate([points, lowered]), band, np.zeros(band.n_vertices),
                             np.concatenate([colours, colours]), camera)]


def device_layers(topology: Topology, vertices: Extensor, semiconductor: Extensor, contacts: list[np.ndarray],
                  camera: Camera) -> list:
    """The device about the silicon: the oxide grey, the contacts' metal filling the oxide's row over the wells, and the
    gate and the body contact as strokes along the border."""
    points = vertices.batch()
    source, drain, body, gate = contacts
    oxide = semiconductor.batch().to_array().reshape(-1) == 0                 # [simplices]
    # The oxide's triangles under the contacts' metal: those whose every corner the source or the drain holds.
    metal = oxide & np.isin(topology.elements[-1], np.concatenate([source, drain])).all(axis=-1)   # [simplices]
    # The gate and the body contact: the border's edges whose ends they hold.
    border = topology.border
    gate_edges, body_edges = (border.subset(np.isin(border.elements[-1], contact).all(axis=-1), -1)
                              for contact in (gate, body))
    n_vertices = topology.n_vertices
    return [contours.picture(points, topology.subset(oxide & ~metal, -1), np.zeros(n_vertices),
                             np.broadcast_to(oxide_colour, (n_vertices, 3)), camera),
            contours.picture(points, topology.subset(metal, -1), np.zeros(n_vertices),
                             np.broadcast_to(metal_colour, (n_vertices, 3)), camera),
            simplices.strokes(points, gate_edges, metal_colour, gate_width, camera),
            simplices.strokes(points, body_edges, metal_colour, body_width, camera)]


def on_white(picture: np.ndarray) -> np.ndarray:
    """Colour and coverage `[rows, columns, 4]` laid over white."""
    return np.clip(picture[..., :3] + (1 - picture[..., 3:]), 0, 1)


@charts.style()
def draw_section(topology: Topology, vertices: Extensor, semiconductor: Extensor, dopants: Extensor,
                 contacts: list[np.ndarray], full: np.ndarray) -> Figure:
    """The MOSFET's cross-section at its vertices `Point-[vertices]`: its silicon tinted by its dopants
    `Point-[vertices]`, the donors' net charge blue and the acceptors' red, as the carriers that would balance them,
    against the full densities `[carriers]`; the oxide, the contacts' metal, the gate and the body contact."""
    mv = vertices.context.multivector
    doping = (mv.t & dopants).batch().to_array().reshape(-1)                  # [vertices]
    figure, axis = charts.subplots(section_width_px, section_height_px)
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
    camera = section_axes(axis, vertices)
    layers = silicon_layers(topology, vertices, semiconductor, np.maximum(doping, 0), np.maximum(-doping, 0), full,
                            camera) + device_layers(topology, vertices, semiconductor, contacts, camera)
    pinned(axis, on_white(drawing.composited(layers)), (*axis.get_xlim(), *axis.get_ylim()))
    return figure


def outputs_axes(axis: plt.Axes, largest_drain: float, current_window: tuple[float, float], volt: float) -> None:
    """The output curves' panel: the drain's current up, without ticks, against its voltage across, in thermal units,
    labelled in volts, `volt` thermal units each, axes through nothing."""
    axis.set(xlim=(0, (1 + drain_room) * largest_drain), ylim=current_window, yticks=[])
    ticks = np.arange(drain_tick_volts, largest_drain / volt, drain_tick_volts)
    axis.set_xticks(ticks * volt, [f"{tick:.1f}" for tick in ticks])
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines["bottom"].set_position(("data", 0.0))
    axis.annotate(r"$V_D$", (1, 0), xycoords=("axes fraction", "data"), xytext=(0, 4), textcoords="offset pixels",
                  ha="right", va="bottom")
    axis.annotate(r"$I_D$", (0, 1), xycoords=("data", "axes fraction"), xytext=(5, 0), textcoords="offset pixels",
                  ha="left", va="top")


def animate_switching(topology: Topology, vertices: Extensor, semiconductor: Extensor, contacts: list[np.ndarray],
                      densities_of: Callable[[Extensor], Extensor],
                      terminal_of: Callable[[Extensor], tuple[Extensor, Extensor]], full: np.ndarray,
                      curves: np.ndarray, largest_drain: float, current_window: tuple[float, float], volt: float,
                      states: Iterable[Extensor]) -> Iterator[np.ndarray]:
    """The MOSFET's cross-section at each state as it comes, its silicon tinted by the electrons and the holes
    `[carriers] Scalar+[vertices]` that `densities_of` reads off it, against their full densities `[carriers]`; and
    beside it the drain's current against its voltage, read off the state by `terminal_of`, the frames on each output
    curve `[frames]`, minus one on none, traced in the curve's colour, in these windows of drain voltage and current."""
    with charts.style():
        figure, (section_axis, outputs_axis) = charts.subplots(
            section_width_px + outputs_width_px, section_height_px, ncols=2,
            gridspec_kw=dict(width_ratios=(section_width_px, outputs_width_px)))
        figure.subplots_adjust(left=0.0, right=0.97, bottom=0.12, top=0.95, wspace=0.12)
        camera = section_axes(section_axis, vertices)
        outputs_axes(outputs_axis, largest_drain, current_window, volt)
        traces = [outputs_axis.plot([], [], color=colour)[0] for colour in curve_colours]
        dot, = outputs_axis.plot([], [], "o", color=dot_colour)
        device = device_layers(topology, vertices, semiconductor, contacts, camera)
        shown = pinned(section_axis, np.ones(camera.shape + (3,)), (*section_axis.get_xlim(),
                                                                    *section_axis.get_ylim()))
        readings = []
        for state in states:
            electrons, holes = densities_of(state).batch().to_array().reshape(2, -1)
            silicon = silicon_layers(topology, vertices, semiconductor, electrons, holes, full, camera)
            shown.set_data(on_white(drawing.composited(silicon + device)))
            readings.append([float(reading.to_array()) for reading in terminal_of(state)])
            # Each curve traced through the frames on it so far.
            traced = np.array(readings)                                       # [frames so far, 2]
            for curve, trace in enumerate(traces):
                trace.set_data(*traced[curves[:len(readings)] == curve].T)
            dot.set_data(*traced[-1:].T)
            yield capture(figure)
        plt.close(figure)


# --- the solar cell ---------------------------------------------------------------------
# The silicon's tint where the light makes the most pairs, fading to white where it makes none; the top layer's tint and
# the back layer's, each over the light's, and how much they cover.
light_tint = np.array(to_rgb("#f2c14e"))
top_tint, back_tint, layer_cover = np.array(to_rgb("#d99a9a")), np.array(to_rgb("#7fa7d1")), 0.4
# The holes' streaks: their colour, how much of their ink shows, and their length in pixels at the reference flow.
streak_colour, streak_ink, cell_streak_length = np.array(to_rgb("#1d2228")), 0.45, 50.0
# The fingers' and the back contact's strokes, in pixels, and the room above the surface for the fingers, as a share of
# the depth.
finger_stroke, back_stroke, finger_room = 6.0, 3.0, 0.05
# The windows of the cell, each about the middle of a gap between two fingers: their pixels across, and the cell's
# window's half width before its fingers spread, in Debye lengths; the window they spread in reaches past the two
# fingers beside its middle at their widest spacing by this factor of their distance from it. A chart's pixels across
# beside a window; and the period alone, its pixels across.
window_width_px, window_half_width, spreading_room = 400, 160.0, 1.08
cell_chart_px, period_width_px = 240, 400
# The current's chart: its voltages' ticks, in volts apart; the power's chart's pixels; the factor its range of
# spacings reaches past the spacings either way, and the factor the charts' ranges reach past what they show.
cell_tick_volts = 0.2
power_size, spacing_margin, chart_margin = (320, 220), 1.08, 1.12


def copies(vertices: Extensor, finger_spacing: float, middle: float, half_width: float) -> list[Extensor]:
    """The vertices of the periods tiling a window this far either way of its middle, each one finger spacing on from
    the one before, `[copies] Point-[vertices]`, the window's middle at nothing and `middle` across from a finger's
    middle: the period is mirrored at its fingers' middles and at its own, so that its mirror image across a finger is
    itself, a spacing on."""
    mv = vertices.context.multivector
    first = int(np.floor((middle - half_width) / finger_spacing))
    last = int(np.ceil((middle + half_width) / finger_spacing)) - 1
    return [vertices + (copy * finger_spacing - middle) * mv.x.dual() for copy in range(first, last + 1)]


def framed(axis: plt.Axes, vertices: Extensor, left: float, right: float) -> Camera:
    """The panel over the solar cell from `left` to `right` across, read by the planes x and y over the weight the plane
    w reads, its whole depth in view with room above for the fingers, and the camera that draws into it at its own
    pixels."""
    mv = vertices.context.multivector
    up = (mv.y & vertices.batch()).to_array()
    room = finger_room * (up.max() - up.min())
    axis.set(xlim=(left, right), ylim=(up.min() - room, up.max() + room))
    axis.set_aspect("equal")
    axis.set_axis_off()
    return panel(axis, mv.x, mv.y, mv.w)


def at_vertices(topology: Topology, values: Extensor) -> Extensor:
    """Values in the simplices `[simplices]` at the vertices `[vertices]`: the mean over the simplices around each."""
    mv = values.context.multivector
    around = topology.incidence[-1, 0].sum(axis=0).adjugate()                 # Scalar+[vertices <- simplices]
    counts = around * mv.scalar(np.ones((topology.n_elements[-1], 1))).field()   # Scalar+[vertices]
    return (around * values) / counts


def compressed(flow: Extensor, reference_flow: float) -> Extensor:
    """A flow `Point-[vertices]` along its own directions, its size the reference flow times the square root of its
    own over the reference's, so that streaks as long as that size show a flow ten times as strong three times as
    long: the holes' flow along the thin top layer, which gathers the flow of the base beneath it, and in the base
    alike. The size is read by the camera's planes x and y."""
    mv = flow.context.multivector
    planes = stack((mv.x, mv.y))                                              # [planes] Plane+
    # The flow's part in space: what each plane reads of it, along the direction that plane reads as one. The flow
    # also runs along time and the carriers' counts, which no plane of the camera reads.
    readings = planes & flow.batch()[..., None]                               # [vertices, planes] Scalar+
    spatial = (readings * planes.dual()).sum(axis=-1)                         # [vertices] Point-
    sizes = spatial.antinorm().to_array().reshape(-1)                         # [vertices]
    scales = np.sqrt(reference_flow / np.maximum(sizes, 1e-12 * reference_flow))   # [vertices]
    return mv.scalar(scales[:, None]).field() * flow


def cell_tints(topology: Topology, vertices: Extensor, dopants: Extensor, generation: Extensor,
               full_generation: float, camera: Camera) -> list:
    """A period of the solar cell at its vertices `Point-[vertices]` as layers: its silicon tinted by the light's
    generation `Scalar+[vertices]`, full at `full_generation`; its top layer, where the dopants `Point-[vertices]` are
    acceptors, and its back layer, where they are many donors, bands of their own tints over it."""
    mv = vertices.context.multivector
    points = vertices.batch()
    shares = np.clip(generation.batch().to_array().reshape(-1) / full_generation, 0, 1)   # [vertices]
    colours = 1 - shares[:, None] * (1 - light_tint)                          # [vertices, 3]
    doping = (mv.t & dopants).batch().to_array().reshape(-1)                  # [vertices]
    # Below nothing the top layer's acceptors, above two the back layer's donors, the base's between.
    layers = contours.Bands(-2.0, 2.0, np.array([np.append(top_tint, layer_cover), np.zeros(4),
                                                 np.append(back_tint, layer_cover)]))
    return [contours.picture(points, topology, np.zeros(topology.n_vertices), colours, camera),
            contours.picture(points, topology, doping, np.zeros(3), camera, bands=(layers,), opacity=0.0)]


def cell_metal(topology: Topology, vertices: Extensor, contacts: list[np.ndarray], camera: Camera) -> list:
    """A period's contacts at its vertices `Point-[vertices]` as layers: the fingers raised on the surface, and the
    back contact along the back."""
    mv = vertices.context.multivector
    points = vertices.batch()
    finger, back = contacts
    border = topology.border
    finger_edges, back_edges = (border.subset(np.isin(border.elements[-1], contact).all(axis=-1), -1)
                                for contact in (finger, back))
    # The fingers stand on the surface: their strokes raised by half their width.
    raised = points + (finger_stroke / 2) * camera.pixel * mv.y.dual()
    return [simplices.strokes(raised, finger_edges, metal_colour, finger_stroke, camera, cap=simplices.Cap.BUTT),
            simplices.strokes(points, back_edges, metal_colour, back_stroke, camera)]


def streaks(topology: Topology, vertices: Extensor, holes: Extensor, reference_flow: float,
            camera: Camera) -> np.ndarray:
    """The holes' flow in each simplex `Point-[simplices]` of a period at its vertices `Point-[vertices]`, streaked
    along its directions, `cell_streak_length` pixels long at the reference flow, longer as the square root of the
    flow: colour and coverage `[rows, columns, 4]`."""
    field = compressed(at_vertices(topology, holes), reference_flow).batch()   # Point-[vertices]
    return flow.layer(vertices.batch(), topology, field, camera, reference_flow, streak_colour,
                      length=cell_streak_length) * np.array([1, 1, 1, streak_ink])


def window_picture(topology: Topology, vertices: Extensor, dopants: Extensor, contacts: list[np.ndarray],
                   generation: Extensor, full_generation: float, holes: Extensor, reference_flow: float,
                   finger_spacing: float, half_width: float, camera: Camera) -> np.ndarray:
    """The window this far either way of the middle of a gap tiled by the periods at their vertices `Point-[vertices]`,
    its fingers this far apart: each period tinted as `cell_tints` tints it, the holes' flow `Point-[simplices]`
    streaked over it, and its contacts over all: colour `[rows, columns, 3]`."""
    tiles = copies(vertices, finger_spacing, finger_spacing / 2, half_width)
    tints = [layer for tile in tiles for layer in cell_tints(topology, tile, dopants, generation, full_generation,
                                                             camera)]
    flows = [streaks(topology, tile, holes, reference_flow, camera) for tile in tiles]
    metal = [layer for tile in tiles for layer in cell_metal(topology, tile, contacts, camera)]
    return on_white(drawing.composited(tints + flows + metal))


@charts.style()
def draw_cell(topology: Topology, vertices: Extensor, dopants: Extensor, contacts: list[np.ndarray],
              generation: Extensor, finger_spacing: float) -> Figure:
    """The solar cell before anything flows: the window about the middle of a gap tiled by its periods at their
    vertices `Point-[vertices]`, its fingers this far apart, each period tinted as `cell_tints` tints it, with its
    contacts."""
    mv = vertices.context.multivector
    up = (mv.y & vertices.batch()).to_array()
    height_px = round(window_width_px * (1 + 2 * finger_room) * np.ptp(up) / (2 * window_half_width))
    figure, axis = charts.subplots(window_width_px, height_px)
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
    camera = framed(axis, vertices, -window_half_width, window_half_width)
    full_generation = float(generation.batch().to_array().max())
    tiles = copies(vertices, finger_spacing, finger_spacing / 2, window_half_width)
    layers = [layer for tile in tiles for layer in cell_tints(topology, tile, dopants, generation, full_generation,
                                                              camera) + cell_metal(topology, tile, contacts, camera)]
    pinned(axis, on_white(drawing.composited(layers)), (*axis.get_xlim(), *axis.get_ylim()))
    return figure


@charts.style()
def draw_period(topology: Topology, vertices: Extensor, dopants: Extensor, contacts: list[np.ndarray],
                generation: Extensor, holes: Extensor, reference_flow: float, finger_spacing: float) -> Figure:
    """One period of the solar cell, its fingers this far apart, from the middle of one gap between them to the
    middle of the next, its finger in the middle: the periods at their vertices `Point-[vertices]` tiling it, tinted
    as `cell_tints` tints them, the holes' flow `Point-[simplices]` streaked over them, and their contacts."""
    mv = vertices.context.multivector
    up = (mv.y & vertices.batch()).to_array()
    half = finger_spacing / 2
    height_px = round(period_width_px * (1 + 2 * finger_room) * np.ptp(up) / finger_spacing)
    figure, axis = charts.subplots(period_width_px, height_px)
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
    camera = framed(axis, vertices, -half, half)
    full_generation = float(generation.batch().to_array().max())
    tiles = copies(vertices, finger_spacing, 0.0, half)
    picture = drawing.composited(
        [layer for tile in tiles for layer in cell_tints(topology, tile, dopants, generation, full_generation, camera)]
        + [streaks(topology, tile, holes, reference_flow, camera) for tile in tiles]
        + [layer for tile in tiles for layer in cell_metal(topology, tile, contacts, camera)])
    pinned(axis, on_white(picture), (*axis.get_xlim(), *axis.get_ylim()))
    return figure


def cell_figure(vertices: Extensor, half_width: float) -> tuple[plt.Figure, Camera, plt.Axes, plt.Axes]:
    """The figure of the solar cell's window this far either way of its middle, its panel as high as the window at the
    panel's width, the camera that draws into the panel, the panel, and a chart beside it."""
    mv = vertices.context.multivector
    up = (mv.y & vertices.batch()).to_array()
    width_px = window_width_px + cell_chart_px
    figure, (cell_axis, chart_axis) = charts.subplots(width_px, window_width_px, ncols=2,
                                                      gridspec_kw=dict(width_ratios=(window_width_px, cell_chart_px)))
    figure.subplots_adjust(left=0.0, right=0.97, bottom=0.1, top=0.95, wspace=0.08)
    # The panel keeps its shares of the figure's width and height: the figure is as high as makes the panel's height
    # the window's at the panel's width.
    panel = cell_axis.get_position()
    height_px = round(width_px * panel.width * (1 + 2 * finger_room) * np.ptp(up) / (2 * half_width) / panel.height)
    figure.set_size_inches(width_px / charts.DPI, height_px / charts.DPI)
    return figure, framed(cell_axis, vertices, -half_width, half_width), cell_axis, chart_axis


def current_axes(axis: plt.Axes, voltage_window: tuple[float, float], current_window: tuple[float, float],
                 volt: float) -> None:
    """The current's panel: the current the cell delivers up, without ticks, against its voltage across, in thermal
    units, labelled in volts, `volt` thermal units each, axes through nothing."""
    axis.set(xlim=voltage_window, ylim=current_window, yticks=[])
    ticks = np.arange(cell_tick_volts, voltage_window[1] / volt, cell_tick_volts)
    axis.set_xticks(ticks * volt, [f"{tick:.1f}" for tick in ticks])
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines["bottom"].set_position(("data", 0.0))
    axis.spines["left"].set_position(("data", 0.0))
    axis.annotate("V", (1, 0), xycoords=("axes fraction", "data"), xytext=(0, 4), textcoords="offset pixels",
                  ha="right", va="bottom")
    axis.annotate("I", (0, 1), xycoords=("data", "axes fraction"), xytext=(5, 0), textcoords="offset pixels",
                  ha="left", va="top")


def spacing_axes(axis: plt.Axes, finger_spacings: np.ndarray, power_window: tuple[float, float]) -> None:
    """The power's panel: the power up against the fingers' spacing across, on a logarithmic scale, axes through
    nothing and without ticks."""
    axis.set(xscale="log", xlim=(finger_spacings[0] / spacing_margin, finger_spacings[-1] * spacing_margin),
             ylim=power_window)
    axis.set_xticks([], minor=False)
    axis.set_xticks([], minor=True)
    axis.set_yticks([])
    axis.spines[["top", "right"]].set_visible(False)
    axis.annotate("spacing", (1, 0), xycoords="axes fraction", xytext=(0, 4), textcoords="offset pixels",
                  ha="right", va="bottom")
    axis.annotate("power", (0, 1), xycoords="axes fraction", xytext=(5, 0), textcoords="offset pixels",
                  ha="left", va="top")


def animate_lighting(topology: Topology, vertices: Extensor, dopants: Extensor, contacts: list[np.ndarray],
                     generation: Extensor, holes_of: Callable[[Extensor], Extensor],
                     terminal_of: Callable[[Extensor], tuple[Extensor, Extensor]], reference_flow: float,
                     finger_spacing: float, highest_bias: float, full_current: Extensor, volt: float,
                     shares: np.ndarray, states: Iterable[Extensor]) -> Iterator[np.ndarray]:
    """The solar cell at its vertices `Point-[vertices]` at each of its potentials as they come, its fingers this far
    apart, under each frame's share `[frames]` of the light's full generation `Scalar+[vertices]`: the window tiled by
    its periods, as `window_picture` draws it, tinted by the frame's light, the holes' flow read off the potentials by
    `holes_of`; and beside it the current it delivers against its voltage `[] Scalar+` each, both read off the
    potentials by `terminal_of`, traced as it goes, the voltage to a little past the highest bias, in thermal units,
    labelled in volts, and the current to a little past the full current `[] Scalar+`."""
    full_generation = float(generation.batch().to_array().max())
    largest = float(full_current.to_array())
    with charts.style():
        figure, camera, cell_axis, curve_axis = cell_figure(vertices, window_half_width)
        current_axes(curve_axis, (0.0, chart_margin * highest_bias), ((1 - chart_margin) * largest,
                                                                      chart_margin * largest), volt)
        trace, = curve_axis.plot([], [], color=lit_colour)
        # The dot whole at no bias, on the chart's edge.
        dot, = curve_axis.plot([], [], "o", color=dot_colour, clip_on=False)
        shown = pinned(cell_axis, np.ones(camera.shape + (3,)), (*cell_axis.get_xlim(), *cell_axis.get_ylim()))
        readings = []
        for share, potentials in zip(shares, states):
            shown.set_data(window_picture(topology, vertices, dopants, contacts, share * generation, full_generation,
                                          holes_of(potentials), reference_flow, finger_spacing, window_half_width,
                                          camera))
            readings.append([float(reading.to_array()) for reading in terminal_of(potentials)])
            trace.set_data(*np.transpose(readings))
            dot.set_data([readings[-1][0]], [readings[-1][1]])
            yield capture(figure)
        plt.close(figure)


def animate_spreading(topology: Topology, dopants: Extensor, contacts: list[np.ndarray], generation: Extensor,
                      holes_of: Callable[[Extensor, Extensor], Extensor],
                      power_of: Callable[[float, Extensor, Extensor], Extensor], reference_flow: float,
                      finger_spacings: np.ndarray, full_power: Extensor,
                      states: Iterable[tuple[Extensor, Extensor]]) -> Iterator[np.ndarray]:
    """The solar cell at each state as it comes, its vertices and its potentials at each of these spacings of its
    fingers `[frames]`: the window tiled by its periods, as `window_picture` draws it, wide enough to hold the two
    fingers beside its middle at the widest spacing, the holes' flow read off the state by `holes_of`; and beside it
    its power per length of surface `[] Scalar+` against the spacing, read off the spacing and the state by
    `power_of`, traced as it goes, to a little past the full power `[] Scalar+`, the most so far marked."""
    full_generation = float(generation.batch().to_array().max())
    states = iter(states)
    first = next(states)
    half_width = spreading_room * finger_spacings[-1] / 2
    with charts.style():
        # The window's depth is the same at every spacing: the first state's vertices frame it.
        figure, camera, cell_axis, power_axis = cell_figure(first[0], half_width)
        spacing_axes(power_axis, finger_spacings, (0.0, chart_margin * float(full_power.to_array())))
        trace, = power_axis.plot([], [], color=lit_colour)
        # The most power so far, a ring under the dot until the power falls past it, and left at the best spacing.
        best, = power_axis.plot([], [], "o", color=dot_colour, fillstyle="none")
        dot, = power_axis.plot([], [], "o", color=dot_colour)
        shown = pinned(cell_axis, np.ones(camera.shape + (3,)), (*cell_axis.get_xlim(), *cell_axis.get_ylim()))
        powers = []
        for finger_spacing, (vertices, potentials) in zip(finger_spacings, chain([first], states)):
            shown.set_data(window_picture(topology, vertices, dopants, contacts, generation, full_generation,
                                          holes_of(vertices, potentials), reference_flow, finger_spacing, half_width,
                                          camera))
            powers.append(float(power_of(finger_spacing, vertices, potentials).to_array()))
            trace.set_data(finger_spacings[:len(powers)], powers)
            most = int(np.argmax(powers))
            best.set_data([finger_spacings[most]], [powers[most]])
            dot.set_data([finger_spacing], [powers[-1]])
            yield capture(figure)
        plt.close(figure)


@charts.style()
def draw_power(finger_spacings: np.ndarray, powers: Extensor) -> Figure:
    """The power the solar cell delivers per length of surface at each spacing of its fingers `[spacings] Scalar+`,
    against the spacing on a logarithmic scale, the highest marked by a ring, as the fingers' spreading marks it."""
    figure, axis = charts.subplots(*power_size)
    powers = powers.to_array().reshape(-1)                                    # [spacings]
    axis.plot(finger_spacings, powers, color=lit_colour)
    best = powers.argmax()
    axis.plot([finger_spacings[best]], [powers[best]], "o", color=dot_colour, fillstyle="none")
    spacing_axes(axis, finger_spacings, (0.0, chart_margin * powers.max()))
    figure.tight_layout(pad=0.4)
    return figure
