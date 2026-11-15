"""Band diagrams, GPU-shaded planar cross-sections, flow streaks, and terminal charts depict the p-n junction, the
planar MOSFET, and the front-contact solar cell.

Along the one-dimensional bar, `band_panel` plots the conduction and valence band edges together with both quasi-Fermi
levels while tinting each band by carrier density, accompanied during bias sweeps by the current-voltage trace. Across
the MOSFET cross-section, GPU triangle shading tints silicon by electron and hole density beneath the grey gate oxide
and dark metal contacts while tracing drain output curves beside it. Across a window of repeating solar-cell periods,
GPU shading shows optical pair generation, emitter and back-surface doping bands, metal contacts, and planar hole flow
streaks alongside either the illuminated current-voltage curve or the delivered power across finger spacings.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure

from examples.util import charts
from examples.util.animation import capture
from examples.util.panels import panel, pinned
from numga import concatenate, stack
from numga.extensor import Extensor
from numga.render import contours, drawing, flow, simplices
from numga.render.camera import Camera
from numga.topology import Topology

# Band-edge, carrier, junction-line, and dark/lit I-V curve colours, with band tints fading to white at intrinsic density.
edge_colour, electron_colour, hole_colour = "#2b2f33", "#2166ac", "#b2182b"
conduction_tint, valence_tint = np.array(to_rgb("#c4d6ec")), np.array(to_rgb("#efcdcd"))
junction_colour = "#c9ced3"
dark_colour, lit_colour = "#2b2f33", "#e08214"
# Alternating dashes keep both quasi-Fermi levels visible where they coincide in equilibrium.
dash = 3.0
electron_dashes, hole_dashes = (0.0, (dash, dash)), (dash, (dash, dash))
edge_width = 1.5 * charts.LINE_WIDTH
band_width_px, band_height_px = 340, 240
curve_width_px = 240
label_room, tick_volts, sweep_room = 0.16, 0.2, 0.08


def band_panel(axis: plt.Axes, places: np.ndarray, window: tuple[float, float],
               intrinsic: float) -> Callable[[np.ndarray, np.ndarray], None]:
    """Configures `axis` over the bar positions (`[V]`) and energy `window` and returns an updater for the four band
    and quasi-Fermi curves (`[4, V]`) and logarithmic carrier-density fills (`[2, V]`)."""
    n_places = len(places)
    left, right = places[0], places[-1]
    axis.set(xlim=(left - label_room * (right - left), right + label_room * (right - left)), ylim=window, xticks=[],
             yticks=[])
    axis.spines[["top", "right", "bottom", "left"]].set_visible(False)
    axis.plot([0.0, 0.0], window, color=junction_colour, linewidth=charts.FRAME_WIDTH, zorder=0)
    for side, place in (("p", left / 2), ("n", right / 2)):
        axis.text(place, window[1], side, ha="center", va="top", color=edge_colour)
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

    def update(energies: np.ndarray, densities: np.ndarray) -> None:
        for band, edge, end, density, tint in zip(bands, energies[:2], window[::-1], densities,
                                                  (conduction_tint, valence_tint)):
            shares = np.clip(np.log((density[:-1] + density[1:]) / 2 / intrinsic) / np.log(1 / intrinsic), 0, 1)
            band.set_verts(np.stack([np.stack([places[:-1], edge[:-1]], -1), np.stack([places[1:], edge[1:]], -1),
                                     np.stack([places[1:], np.full(n_places - 1, end)], -1),
                                     np.stack([places[:-1], np.full(n_places - 1, end)], -1)], axis=1))
            band.set_facecolor(1 - shares[:, None] * (1 - tint))
        for line, energy in zip((conduction, valence, electrons, holes), energies):
            line.set_data(places, energy)
        margin = 0.02 * (right - left)
        for label, energy in zip(labels[:3], energies[:3]):
            label.set_position((right + margin, energy[-1]))
        labels[3].set_position((left - margin, energies[3][0]))

    return update


def bands_figure(vertices: Extensor, energies: Extensor, densities: Extensor, intrinsic: float,
                 window: tuple[float, float]) -> plt.Figure:
    """Band and quasi-Fermi `energies` (`[4] Scalar+[V]`) and carrier `densities` (`[2] Scalar+[V]`) form a still band
    diagram over `vertices` (`Point-[V]`)."""
    mv = vertices.context.multivector
    places = np.asarray((mv.x & vertices.batch()).to_array())
    with charts.style():
        figure, axis = charts.subplots(band_width_px, band_height_px)
        figure.subplots_adjust(left=0.0, right=1.0, bottom=0.02, top=0.98)
        band_panel(axis, places, window, intrinsic)(np.asarray(energies.batch().to_array()).reshape(4, -1),
                                                    np.asarray(densities.batch().to_array()).reshape(2, -1))
    return figure


def formation_animation(vertices: Extensor, bands_of: Callable[[Extensor], Extensor],
                      carriers_of: Callable[[Extensor], Extensor], intrinsic: float, window: tuple[float, float],
                      states: Iterable[Extensor]) -> Iterator[np.ndarray]:
    """Yields band-diagram frames over `vertices` (`Point-[V]`) as `states` evolve toward junction equilibrium."""
    mv = vertices.context.multivector
    places = np.asarray((mv.x & vertices.batch()).to_array())
    with charts.style():
        figure, axis = charts.subplots(band_width_px, band_height_px)
        figure.subplots_adjust(left=0.0, right=1.0, bottom=0.02, top=0.98)
        update = band_panel(axis, places, window, intrinsic)
        for state in states:
            update(np.asarray(bands_of(state).batch().to_array()).reshape(4, -1),
                   np.asarray(carriers_of(state).batch().to_array()).reshape(2, -1))
            yield capture(figure)
        plt.close(figure)


def sweep_animation(vertices: Extensor, bands_of: Callable[[Extensor], Extensor],
                  carriers_of: Callable[[Extensor], Extensor], intrinsic: float,
                  terminal_of: Callable[[Extensor], tuple[Extensor, Extensor]], window: tuple[float, float],
                  voltages: tuple[float, float], currents: tuple[float, float], volt: float, colour: str,
                  reference_sweeps: list[list[Extensor]], states: Iterable[Extensor]) -> Iterator[np.ndarray]:
    """Yields side-by-side band-diagram and current-voltage frames across `states`, drawing any `reference_sweeps`
    faintly beneath the active trace."""
    mv = vertices.context.multivector
    places = np.asarray((mv.x & vertices.batch()).to_array())
    with charts.style():
        figure, (band_axis, curve_axis) = charts.subplots(
            band_width_px + curve_width_px, band_height_px, ncols=2,
            gridspec_kw=dict(width_ratios=(band_width_px, curve_width_px)))
        figure.subplots_adjust(left=0.0, right=0.97, bottom=0.1, top=0.92, wspace=0.08)
        update = band_panel(band_axis, places, window, intrinsic)
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
        for prior_sweep in reference_sweeps:
            curve = [[float(reading.to_array()) for reading in terminal_of(state)] for state in prior_sweep]
            curve_axis.plot(*np.transpose(curve), color=dark_colour, alpha=0.35)
        trace, = curve_axis.plot([], [], color=colour)
        dot, = curve_axis.plot([], [], "o", color=colour)
        trace_points = []
        for state in states:
            update(np.asarray(bands_of(state).batch().to_array()).reshape(4, -1),
                   np.asarray(carriers_of(state).batch().to_array()).reshape(2, -1))
            voltage, current = terminal_of(state)
            trace_points.append((float(voltage.to_array()), float(current.to_array())))
            trace.set_data(*np.transpose(trace_points))
            dot.set_data([trace_points[-1][0]], [trace_points[-1][1]])
            yield capture(figure)
        plt.close(figure)


# --- MOSFET cross-section -------------------------------------------------------------
electron_tint, hole_tint = np.array(to_rgb("#3a78b8")), np.array(to_rgb("#dca2a2"))
oxide_colour, metal_colour = np.array(to_rgb("#d4d7da")), np.array(to_rgb("#3b4a57"))
surface_width, gate_width, body_width = 4.0, 4.0, 3.0
curve_colours = ("#9ecae1", "#4292c6", "#08306b")
dot_colour, drain_tick_volts, drain_room = "#2b2f33", 0.5, 0.04
section_width_px, section_height_px, section_margin = 360, 200, 0.02
outputs_width_px = 250


def section_axes(axis: plt.Axes, vertices: Extensor) -> Camera:
    """Frames `axis` around `vertices` (`Point-[V]`) with equal aspect and returns a planar camera weighed by `w`."""
    mv = vertices.context.multivector
    points = vertices.batch()
    x_coords, y_coords = (stack((mv.x, mv.y)) & points[:, None]).to_array().T
    margin = section_margin * (x_coords.max() - x_coords.min())
    axis.set(xlim=(x_coords.min() - margin, x_coords.max() + margin),
             ylim=(y_coords.min() - margin, y_coords.max() + margin))
    axis.set_aspect("equal")
    axis.set_axis_off()
    return panel(axis, mv.x, mv.y, mv.w)


def tints(electrons: np.ndarray, holes: np.ndarray, full: np.ndarray) -> np.ndarray:
    """Carrier fractions relative to `full` (`[2]`) blend electron blue and hole red on white (`[..., 3]`)."""
    electron_share, hole_share = np.clip(electrons / full[0], 0, 1), np.clip(holes / full[1], 0, 1)
    return np.clip(1 - electron_share[..., None] * (1 - electron_tint) - hole_share[..., None] * (1 - hole_tint), 0, 1)


def silicon_layers(topology: Topology, vertices: Extensor, semiconductor: Extensor, electrons: np.ndarray,
                   holes: np.ndarray, full: np.ndarray, camera: Camera) -> list:
    """Renders the carrier-tinted silicon triangles and a pixel-thick surface band beneath the gate oxide so the thin
    inversion channel remains visible."""
    V, E, F = topology.n_elements
    mv = vertices.context.multivector
    points = vertices.batch()
    silicon = semiconductor.batch().to_array().reshape(-1) > 0                # [F]
    border = topology.subset(silicon, -1).border
    surface = border.subset(np.isin(border.elements[-1], topology.elements[-1][~silicon]).all(axis=-1), -1)
    colours = tints(electrons, holes, full)                                   # [V, 3]
    band = surface.product(Topology.chains([2]))
    V_band, E_band, F_band = band.n_elements
    band_bottom = points - surface_width * camera.pixel * mv.y.dual()
    return [contours.picture(points, topology.subset(silicon, -1), np.zeros(V), colours, camera),
            contours.picture(concatenate([points, band_bottom]), band, np.zeros(V_band),
                             np.concatenate([colours, colours]), camera)]


def device_layers(topology: Topology, vertices: Extensor, semiconductor: Extensor, contacts: list[np.ndarray],
                  camera: Camera) -> list:
    """Renders the grey gate oxide, the source and drain metal plugs through the oxide row, and the gate and body
    boundary strokes."""
    V, E, F = topology.n_elements
    points = vertices.batch()
    source, drain, body, gate = contacts
    oxide = semiconductor.batch().to_array().reshape(-1) == 0                 # [F]
    metal = oxide & np.isin(topology.elements[-1], np.concatenate([source, drain])).all(axis=-1)   # [F]
    border = topology.border
    gate_edges, body_edges = (border.subset(np.isin(border.elements[-1], contact).all(axis=-1), -1)
                              for contact in (gate, body))
    return [contours.picture(points, topology.subset(oxide & ~metal, -1), np.zeros(V),
                             np.broadcast_to(oxide_colour, (V, 3)), camera),
            contours.picture(points, topology.subset(metal, -1), np.zeros(V),
                             np.broadcast_to(metal_colour, (V, 3)), camera),
            simplices.strokes(points, gate_edges, metal_colour, gate_width, camera),
            simplices.strokes(points, body_edges, metal_colour, body_width, camera)]


def white_composite(picture: np.ndarray) -> np.ndarray:
    """Premultiplied RGBA `picture` (`[..., 4]`) composites over a white background (`[..., 3]`)."""
    return np.clip(picture[..., :3] + (1 - picture[..., 3:]), 0, 1)


@charts.style()
def section_figure(topology: Topology, vertices: Extensor, semiconductor: Extensor, dopants: Extensor,
                   contacts: list[np.ndarray], full: np.ndarray) -> Figure:
    """Silicon doping tints from `dopants` (`Point-[V]`) relative to `full` (`[2]`) combine with oxide and contact
    layers into the MOSFET cross-section figure at `vertices` (`Point-[V]`)."""
    V, E, F = topology.n_elements
    mv = vertices.context.multivector
    doping = (mv.t & dopants).batch().to_array().reshape(-1)                  # [V]
    figure, axis = charts.subplots(section_width_px, section_height_px)
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
    camera = section_axes(axis, vertices)
    layers = silicon_layers(topology, vertices, semiconductor, np.maximum(doping, 0), np.maximum(-doping, 0), full,
                            camera) + device_layers(topology, vertices, semiconductor, contacts, camera)
    pinned(axis, white_composite(drawing.composited(layers)), (*axis.get_xlim(), *axis.get_ylim()))
    return figure


def outputs_axes(axis: plt.Axes, largest_drain: float, current_window: tuple[float, float], volt: float) -> None:
    """Configures `axis` for MOSFET drain current against drain voltage labelled in volts."""
    axis.set(xlim=(0, (1 + drain_room) * largest_drain), ylim=current_window, yticks=[])
    ticks = np.arange(drain_tick_volts, largest_drain / volt, drain_tick_volts)
    axis.set_xticks(ticks * volt, [f"{tick:.1f}" for tick in ticks])
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines["bottom"].set_position(("data", 0.0))
    axis.annotate(r"$V_D$", (1, 0), xycoords=("axes fraction", "data"), xytext=(0, 4), textcoords="offset pixels",
                  ha="right", va="bottom")
    axis.annotate(r"$I_D$", (0, 1), xycoords=("data", "axes fraction"), xytext=(5, 0), textcoords="offset pixels",
                  ha="left", va="top")


def transistor_animation(topology: Topology, vertices: Extensor, semiconductor: Extensor, contacts: list[np.ndarray],
                       densities_of: Callable[[Extensor], Extensor],
                       terminal_of: Callable[[Extensor], tuple[Extensor, Extensor]], full: np.ndarray,
                       curves: np.ndarray, largest_drain: float, current_window: tuple[float, float], volt: float,
                       states: Iterable[Extensor]) -> Iterator[np.ndarray]:
    """Yields side-by-side MOSFET cross-section and drain output-curve frames across `states`, colouring each sweep
    indexed by `curves` (`[n_frames]`)."""
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
        image = pinned(section_axis, np.ones(camera.shape + (3,)), (*section_axis.get_xlim(),
                                                                    *section_axis.get_ylim()))
        readings = []
        for state in states:
            electrons, holes = densities_of(state).batch().to_array().reshape(2, -1)
            silicon = silicon_layers(topology, vertices, semiconductor, electrons, holes, full, camera)
            image.set_data(white_composite(drawing.composited(silicon + device)))
            readings.append([float(reading.to_array()) for reading in terminal_of(state)])
            trace_points = np.array(readings)                                 # [..., 2]
            for curve, trace in enumerate(traces):
                trace.set_data(*trace_points[curves[:len(readings)] == curve].T)
            dot.set_data(*trace_points[-1:].T)
            yield capture(figure)
        plt.close(figure)


# --- solar cell -----------------------------------------------------------------------
light_tint = np.array(to_rgb("#f2c14e"))
top_tint, back_tint, layer_cover = np.array(to_rgb("#d99a9a")), np.array(to_rgb("#7fa7d1")), 0.4
streak_colour, streak_ink, cell_streak_length = np.array(to_rgb("#1d2228")), 0.45, 50.0
finger_stroke, back_stroke, finger_room = 6.0, 3.0, 0.05
window_width_px, window_half_width, spreading_room = 400, 160.0, 1.08
cell_chart_px, period_width_px = 240, 400
cell_tick_volts = 0.2
power_size, spacing_margin, chart_margin = (320, 220), 1.08, 1.12


def period_copies(vertices: Extensor, finger_spacing: float, middle: float, half_width: float) -> list[Extensor]:
    """Integer multiples of `finger_spacing` shift `vertices` (`Point-[V]`) horizontally to tile `[-half_width, half_width]`
    around `middle`."""
    mv = vertices.context.multivector
    first = int(np.floor((middle - half_width) / finger_spacing))
    last = int(np.ceil((middle + half_width) / finger_spacing)) - 1
    return [vertices + (copy * finger_spacing - middle) * mv.x.dual() for copy in range(first, last + 1)]


def window_axes(axis: plt.Axes, vertices: Extensor, left: float, right: float) -> Camera:
    """Frames `axis` from `left` to `right` across the full depth of `vertices` (`Point-[V]`) and returns a planar
    camera weighed by `w`."""
    mv = vertices.context.multivector
    y_coords = (mv.y & vertices.batch()).to_array()
    room = finger_room * (y_coords.max() - y_coords.min())
    axis.set(xlim=(left, right), ylim=(y_coords.min() - room, y_coords.max() + room))
    axis.set_aspect("equal")
    axis.set_axis_off()
    return panel(axis, mv.x, mv.y, mv.w)


def vertex_averages(topology: Topology, values: Extensor) -> Extensor:
    """Incident-triangle sums divided by valence average simplex `values` (`...[F]`) onto vertices (`...[V]`)."""
    V, E, F = topology.n_elements
    mv = values.context.multivector
    vertex_incidence = topology.incidence[-1, 0].sum(axis=0).adjugate()       # Scalar+[V <- F]
    counts = vertex_incidence * mv.scalar(np.ones((F, 1))).field()            # Scalar+[V]
    return (vertex_incidence * values) / counts


def scaled_flow(flow_field: Extensor, reference_flow: float) -> Extensor:
    """Square-root scaling of spatial norm relative to `reference_flow` compresses `flow_field` (`Point-[V]`) so both
    base diffusion and emitter sheet flow remain legible."""
    mv = flow_field.context.multivector
    points = flow_field.batch()
    V = len(points)
    planes = stack((mv.x, mv.y))                                              # [2] Plane+
    readings = planes & points[..., None]                                     # [V, 2] Scalar+
    spatial = (readings * planes.dual()).sum(axis=-1)                         # [V] Point-
    sizes = spatial.antinorm().to_array().reshape(-1)                         # [V]
    scales = np.sqrt(reference_flow / np.maximum(sizes, 1e-12 * reference_flow))   # [V]
    return mv.scalar(scales[:, None]).field() * flow_field


def cell_tints(topology: Topology, vertices: Extensor, dopants: Extensor, generation: Extensor,
               full_generation: float, camera: Camera) -> list:
    """Renders optical generation shading (`Scalar+[V]`) together with translucent emitter and back-surface doping
    bands (`Point-[V]`)."""
    V, E, F = topology.n_elements
    mv = vertices.context.multivector
    points = vertices.batch()
    shares = np.clip(generation.batch().to_array().reshape(-1) / full_generation, 0, 1)   # [V]
    colours = 1 - shares[:, None] * (1 - light_tint)                          # [V, 3]
    doping = (mv.t & dopants).batch().to_array().reshape(-1)                  # [V]
    layers = contours.Bands(-2.0, 2.0, np.array([np.append(top_tint, layer_cover), np.zeros(4),
                                                 np.append(back_tint, layer_cover)]))
    return [contours.picture(points, topology, np.zeros(V), colours, camera),
            contours.picture(points, topology, doping, np.zeros(3), camera, bands=(layers,), opacity=0.0)]


def cell_metal(topology: Topology, vertices: Extensor, contacts: list[np.ndarray], camera: Camera) -> list:
    """Renders the front contact fingers resting on the top surface and the back metal contact along the bottom."""
    mv = vertices.context.multivector
    points = vertices.batch()
    finger, back = contacts
    border = topology.border
    finger_edges, back_edges = (border.subset(np.isin(border.elements[-1], contact).all(axis=-1), -1)
                                for contact in (finger, back))
    finger_points = points + (finger_stroke / 2) * camera.pixel * mv.y.dual()
    return [simplices.strokes(finger_points, finger_edges, metal_colour, finger_stroke, camera, cap=simplices.Cap.BUTT),
            simplices.strokes(points, back_edges, metal_colour, back_stroke, camera)]


def streaks(topology: Topology, vertices: Extensor, holes: Extensor, reference_flow: float,
            camera: Camera) -> np.ndarray:
    """Renders premultiplied RGBA flow streaks (`[..., 4]`) from triangle hole flows (`Point-[F]`) over `vertices`
    (`Point-[V]`)."""
    field = scaled_flow(vertex_averages(topology, holes), reference_flow).batch()
    return flow.layer(vertices.batch(), topology, field, camera, reference_flow, streak_colour,
                      length=cell_streak_length) * np.array([1, 1, 1, streak_ink])


def window_picture(topology: Topology, vertices: Extensor, dopants: Extensor, contacts: list[np.ndarray],
                   generation: Extensor, full_generation: float, holes: Extensor, reference_flow: float,
                   finger_spacing: float, half_width: float, camera: Camera) -> np.ndarray:
    """Composites tiled solar-cell periods, hole flow streaks (`Point-[F]`), and metal contacts across a window of
    `half_width` around the gap midpoint."""
    tiles = period_copies(vertices, finger_spacing, finger_spacing / 2, half_width)
    tint_layers = [layer for tile in tiles
                   for layer in cell_tints(topology, tile, dopants, generation, full_generation, camera)]
    flow_layers = [streaks(topology, tile, holes, reference_flow, camera) for tile in tiles]
    metal_layers = [layer for tile in tiles for layer in cell_metal(topology, tile, contacts, camera)]
    return white_composite(drawing.composited(tint_layers + flow_layers + metal_layers))


@charts.style()
def cell_figure(topology: Topology, vertices: Extensor, dopants: Extensor, contacts: list[np.ndarray],
                generation: Extensor, finger_spacing: float) -> Figure:
    """Tiled periods at `finger_spacing` show the solar-cell geometry, doping bands, and optical `generation` profile
    (`Scalar+[V]`)."""
    mv = vertices.context.multivector
    y_coords = (mv.y & vertices.batch()).to_array()
    height_px = round(window_width_px * (1 + 2 * finger_room) * np.ptp(y_coords) / (2 * window_half_width))
    figure, axis = charts.subplots(window_width_px, height_px)
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
    camera = window_axes(axis, vertices, -window_half_width, window_half_width)
    full_generation = float(generation.batch().to_array().max())
    tiles = period_copies(vertices, finger_spacing, finger_spacing / 2, window_half_width)
    layers = [layer for tile in tiles
              for layer in cell_tints(topology, tile, dopants, generation, full_generation, camera)
              + cell_metal(topology, tile, contacts, camera)]
    pinned(axis, white_composite(drawing.composited(layers)), (*axis.get_xlim(), *axis.get_ylim()))
    return figure


@charts.style()
def period_figure(topology: Topology, vertices: Extensor, dopants: Extensor, contacts: list[np.ndarray],
                  generation: Extensor, holes: Extensor, reference_flow: float, finger_spacing: float) -> Figure:
    """One full solar-cell period centred on a front finger shows optical `generation` (`Scalar+[V]`), doping bands,
    and hole flow streaks (`Point-[F]`)."""
    mv = vertices.context.multivector
    y_coords = (mv.y & vertices.batch()).to_array()
    half = finger_spacing / 2
    height_px = round(period_width_px * (1 + 2 * finger_room) * np.ptp(y_coords) / finger_spacing)
    figure, axis = charts.subplots(period_width_px, height_px)
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
    camera = window_axes(axis, vertices, -half, half)
    full_generation = float(generation.batch().to_array().max())
    tiles = period_copies(vertices, finger_spacing, 0.0, half)
    picture = drawing.composited(
        [layer for tile in tiles for layer in cell_tints(topology, tile, dopants, generation, full_generation, camera)]
        + [streaks(topology, tile, holes, reference_flow, camera) for tile in tiles]
        + [layer for tile in tiles for layer in cell_metal(topology, tile, contacts, camera)])
    pinned(axis, white_composite(picture), (*axis.get_xlim(), *axis.get_ylim()))
    return figure


def window_layout(vertices: Extensor, half_width: float) -> tuple[plt.Figure, Camera, plt.Axes, plt.Axes]:
    """Two-panel layout pairs a solar-cell cross-section window of `half_width` with a companion chart."""
    mv = vertices.context.multivector
    y_coords = (mv.y & vertices.batch()).to_array()
    width_px = window_width_px + cell_chart_px
    figure, (cell_axis, chart_axis) = charts.subplots(width_px, window_width_px, ncols=2,
                                                      gridspec_kw=dict(width_ratios=(window_width_px, cell_chart_px)))
    figure.subplots_adjust(left=0.0, right=0.97, bottom=0.1, top=0.95, wspace=0.08)
    box = cell_axis.get_position()
    height_px = round(width_px * box.width * (1 + 2 * finger_room) * np.ptp(y_coords) / (2 * half_width) / box.height)
    figure.set_size_inches(width_px / charts.DPI, height_px / charts.DPI)
    return figure, window_axes(cell_axis, vertices, -half_width, half_width), cell_axis, chart_axis


def current_axes(axis: plt.Axes, voltage_window: tuple[float, float], current_window: tuple[float, float],
                 volt: float) -> None:
    """Configures `axis` for delivered solar-cell current against terminal voltage labelled in volts."""
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
    """Configures `axis` for delivered power against logarithmic `finger_spacings`."""
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


def lighting_animation(topology: Topology, vertices: Extensor, dopants: Extensor, contacts: list[np.ndarray],
                     generation: Extensor, holes_of: Callable[[Extensor], Extensor],
                     terminal_of: Callable[[Extensor], tuple[Extensor, Extensor]], reference_flow: float,
                     finger_spacing: float, highest_bias: float, full_current: Extensor, volt: float,
                     shares: np.ndarray, states: Iterable[Extensor]) -> Iterator[np.ndarray]:
    """Yields side-by-side solar-cell cross-section and I-V curve frames as illumination `shares` (`[n_frames]`) turn
    on and forward bias rises toward open circuit."""
    full_generation = float(generation.batch().to_array().max())
    largest = float(full_current.to_array())
    with charts.style():
        figure, camera, cell_axis, curve_axis = window_layout(vertices, window_half_width)
        current_axes(curve_axis, (0.0, chart_margin * highest_bias), ((1 - chart_margin) * largest,
                                                                      chart_margin * largest), volt)
        trace, = curve_axis.plot([], [], color=lit_colour)
        dot, = curve_axis.plot([], [], "o", color=dot_colour, clip_on=False)
        image = pinned(cell_axis, np.ones(camera.shape + (3,)), (*cell_axis.get_xlim(), *cell_axis.get_ylim()))
        readings = []
        for share, potentials in zip(shares, states):
            image.set_data(window_picture(topology, vertices, dopants, contacts, share * generation, full_generation,
                                          holes_of(potentials), reference_flow, finger_spacing, window_half_width,
                                          camera))
            readings.append([float(reading.to_array()) for reading in terminal_of(potentials)])
            trace.set_data(*np.transpose(readings))
            dot.set_data([readings[-1][0]], [readings[-1][1]])
            yield capture(figure)
        plt.close(figure)


def finger_sweep_animation(topology: Topology, dopants: Extensor, contacts: list[np.ndarray], generation: Extensor,
                         holes_of: Callable[[Extensor, Extensor], Extensor],
                         power_of: Callable[[float, Extensor, Extensor], Extensor], reference_flow: float,
                         finger_spacings: np.ndarray, full_power: Extensor,
                         states: Iterable[tuple[Extensor, Extensor]]) -> Iterator[np.ndarray]:
    """Yields side-by-side solar-cell cross-section and delivered-power frames as `finger_spacings` (`[n_spacings]`)
    widen across `states`, marking the running maximum power with an open ring."""
    full_generation = float(generation.batch().to_array().max())
    iterator = iter(states)
    first = next(iterator)
    half_width = spreading_room * finger_spacings[-1] / 2
    with charts.style():
        figure, camera, cell_axis, power_axis = window_layout(first[0], half_width)
        spacing_axes(power_axis, finger_spacings, (0.0, chart_margin * float(full_power.to_array())))
        trace, = power_axis.plot([], [], color=lit_colour)
        best, = power_axis.plot([], [], "o", color=dot_colour, fillstyle="none")
        dot, = power_axis.plot([], [], "o", color=dot_colour)
        image = pinned(cell_axis, np.ones(camera.shape + (3,)), (*cell_axis.get_xlim(), *cell_axis.get_ylim()))
        powers = []
        for finger_spacing, (vertices, potentials) in zip(finger_spacings, chain([first], iterator)):
            image.set_data(window_picture(topology, vertices, dopants, contacts, generation, full_generation,
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
def power_figure(finger_spacings: np.ndarray, powers: Extensor) -> Figure:
    """Delivered power per unit surface length (`[n_spacings] Scalar+`) against logarithmic `finger_spacings`
    (`[n_spacings]`) highlights the optimal spacing with an open ring."""
    n_spacings = len(finger_spacings)
    figure, axis = charts.subplots(*power_size)
    power_values = powers.to_array().reshape(n_spacings)                      # [n_spacings]
    axis.plot(finger_spacings, power_values, color=lit_colour)
    best = power_values.argmax()
    axis.plot([finger_spacings[best]], [power_values[best]], "o", color=dot_colour, fillstyle="none")
    spacing_axes(axis, finger_spacings, (0.0, chart_margin * power_values.max()))
    figure.tight_layout(pad=0.4)
    return figure
