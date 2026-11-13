"""Drawing for adiabatic demagnetization in a crystal: the slab tinted by its temperature, a spin at each of its sites,
a dot where it lies along the field and a ring where it lies against it, the magnet's field as a wash of the gap around
it, and the bath and the load as bars at its ends, touching it while linked; beside it the chart of entropy and
temperature, with the line of the magnet's field and the crystal's states on it; and the salt's equation of state and
the ideal cycle on the temperature-entropy chart as charts.

The field and the spins are patches of one plane of space, the slab's own. A spin lies along the field or against it,
read by pairing the magnet's plane with it, `core.magnet_plane & spins`. The field fills the magnet's gap, read by
pairing it with a unit magnetization, `field & spin`, against the highest field. The crystal's places are read by the
planes x and y and weighed by w. Its tint runs evenly in the logarithm of the temperature between fixed limits, blue
through pale yellow to red, mixed with white so that the spins read on it; the chart's entropies and temperatures are
read off the same states, by `mv.s & states` and `potentials & entropy`."""
from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure

from numga import stack
from numga.extensor import Extensor
from examples.util import charts
from examples.util.animation import capture
from numga.topology import Topology
from numga.render import contours, drawing, simplices
from numga.render.camera import Camera
from examples.util.panels import image_figure
from examples.thermodynamics.paramagnetic_salt import core

mv = core.mv
# The crystal's tint, evenly in the logarithm of the temperature between these limits, and its share of white.
temperature_colours = colormaps["RdYlBu_r"]
coldest, warmest, paleness = 3e-4, 5.0, 0.5
# The magnet's gap around the crystal, in the crystal's widths: its colour, its wash's depth at the highest field and
# its outline's width, in pixels.
gap_low, gap_high = np.array([-0.3, -0.22]), np.array([2.3, 1.22])
magnet_colour, magnet_depth, magnet_outline = np.array(to_rgb("#3a9a6b")), 0.6, 1.5
# The crystal's outline and its triangles' edges, their colour and widths, and the reservoirs' bars' width, their
# outline's width, in the outline's colour, and their gap from the crystal while open, in pixels.
outline_colour, outline_width, edge_width = np.array(to_rgb("#3d4148")), 1.5, 1.0
bar_width, bar_outline, bar_gap = 5.0, 1.5, 7.0
# A spin: its dot's colour and radius, and, on one against the field, the paler dot over it and the width of the rim
# that leaves, in pixels.
spin_colour, against_colour, spin_radius, spin_rim = np.array(to_rgb("#2b2140")), np.ones(3), 4.5, 1.7
# The scene's span in the crystal's widths, across and up, and its pixels to a width.
scene_low, scene_high, pixels_per_width = np.array([-0.36, -0.28]), np.array([2.36, 1.28]), 110
# The chart beside the scene: its width in pixels, the ideal cycle's colour on it, the width of the line of the magnet's
# field, the dots of the crystal's states, their size, colour and rim, and the ranges of entropy and temperature shown.
chart_width, ideal_colour, field_line_px = 280, "0.6", 2.5
state_dot, state_colour, state_rim = 7.0, "#1d2228", "white"
chart_entropy_range, chart_temperature_range = (0.0, 0.76), (3e-4, 6.0)
# The equation of state's line's colour, and the temperatures named on the charts.
profile_colour = "#1d2228"
named_temperatures, temperature_names = [1e-3, 1e-2, 1e-1, 1.0], ["1 mK", "10 mK", "100 mK", "1 K"]
# The temperature-entropy chart: the ideal cycle's colour, the lines of equal field's and their names', and their
# offset from the lines' warmest ends; the cycle's and the lines' widths, in pixels; the ranges of entropy and
# temperature shown; and its pixels, and the equation of state's.
cycle_colour, line_colour, name_colour = "#1d2228", "0.75", "0.45"
name_offset = (-4, 0)
cycle_px, line_px = 2.0, 1.0
entropy_range, temperature_range = (0.0, 0.76), (4e-4, 2.4)
cycle_size, state_size = (300, 240), (300, 210)


# --- plumbing -------------------------------------------------------------------------
def tinted(temperatures: np.ndarray) -> np.ndarray:
    """The colours `[..., 3]` of these temperatures `[...]`, evenly in their logarithm between the limits."""
    share = np.log(temperatures / coldest) / np.log(warmest / coldest)
    return np.asarray(temperature_colours(np.clip(share, 0, 1)))[..., :3]


def on_white(picture: np.ndarray) -> np.ndarray:
    """Colour and coverage `[rows, columns, 4]` laid over white, as bytes `[rows, columns, 3]`."""
    shown = picture[..., :3] + (1 - picture[..., 3:])
    return np.round(np.clip(shown, 0, 1) * 255).astype(np.uint8)


def place(across: np.ndarray, up: np.ndarray) -> Extensor:
    """The scene's places this far across and up `[...]`, in the crystal's widths, points of weight one."""
    return (mv.x * across + mv.y * up + mv.w).dual()


def scene_camera() -> Camera:
    """The camera on the scene, reading points by the planes x and y and weighing them by w, over the scene's span."""
    return Camera(mv.x, mv.y, 0 * mv.x, mv.w, 0.0, (scene_low + scene_high) / 2, (scene_high - scene_low) / 2,
                  round((scene_high - scene_low)[0] * pixels_per_width))


def ends(topology: Topology, points: Extensor) -> tuple[list[np.ndarray], list[Topology]]:
    """The crystal's two ends at these points `[vertices] Point-`: the vertices with the least and the most reading
    across `[ends][vertices]`, and the border's edges along each."""
    along = (mv.x & points).to_array().ravel()                                # [vertices]
    end_vertices = [np.flatnonzero(along == along.min()), np.flatnonzero(along == along.max())]
    border = topology.border
    return end_vertices, [border.subset(np.isin(border.elements[1], vertices).all(axis=-1), 1)
                          for vertices in end_vertices]


def bars(points: Extensor, end_edges: list[Topology], linked: np.ndarray, colours: np.ndarray, camera: Camera) -> list:
    """The reservoirs' bars along the crystal's ends' edges, each in its colour `[ends, 3]`, out from the crystal by
    half their width, touching it where `linked` `[ends]` holds and by the gap further out where it does not."""
    outward = stack((-mv.x.dual(), mv.x.dual()))                              # [ends] Point-
    reach = (bar_width / 2 + bar_outline + bar_gap * ~linked) * camera.pixel  # [ends]
    # Each bar over a wider stroke of the outline's colour, which sets it off from white.
    return [layer for edges, direction, distance, colour in zip(end_edges, outward, reach, colours)
            for layer in (simplices.strokes(points + distance * direction, edges, outline_colour,
                                            bar_width + 2 * bar_outline, camera),
                          simplices.strokes(points + distance * direction, edges, colour, bar_width, camera))]


def scene(topology: Topology, points: Extensor, end_vertices: list[np.ndarray], end_edges: list[Topology],
          sites: np.ndarray, reservoir_colours: np.ndarray, highest_field: float, salt_potentials: core.Potential,
          field: core.Magnetic, links: core.Scalar, spins: core.Spin, camera: Camera) -> np.ndarray:
    """The crystal at its points `[vertices] Point-` in one frame, as bytes: the magnet's gap around it, outlined and
    washed, its depth the square root of the field's share of the highest, so that a field of a few hundredths of the
    highest still shows and no field leaves it white; the slab tinted, palely, by its temperatures; the reservoirs'
    bars, touching it while linked; and a spin at each site `[sites]`, a dot where it lies along the field and a ring
    where it lies against it, read by pairing the magnet's plane with the spin."""
    temperatures = (salt_potentials & core.entropy).batch().to_array().ravel()   # [vertices]
    linked = links.batch().to_array().ravel() > 0                             # [vertices]
    against = (core.magnet_plane & spins).batch().to_array().ravel()[sites] < 0   # [sites]
    depth = magnet_depth * np.sqrt(float(((field & core.spin) / highest_field).to_array()))
    corners = place(np.array([gap_low[0], gap_high[0], gap_high[0], gap_low[0]]),
                    np.array([gap_low[1], gap_low[1], gap_high[1], gap_high[1]]))   # [4] Point-
    return on_white(drawing.composited([
        contours.picture(corners, Topology(4, [[0, 1, 2], [0, 2, 3]]), np.zeros(4), 1 - depth * (1 - magnet_colour),
                         camera),
        simplices.strokes(corners, Topology.loops([4]), magnet_colour, magnet_outline, camera),
        contours.picture(points, topology, np.zeros(topology.n_vertices),
                         paleness + (1 - paleness) * tinted(temperatures), camera),
        simplices.strokes(points, topology.border, outline_colour, outline_width, camera),
        *bars(points, end_edges, np.array([linked[vertices].any() for vertices in end_vertices]), reservoir_colours,
              camera),
        simplices.markers(points[sites], spin_colour, spin_radius, camera),
        simplices.markers(points[sites][against], against_colour, spin_radius - spin_rim, camera)]))


def chart(height: int, corner_entropies: core.Scalar, corner_temperatures: core.Scalar, line_states: core.State,
          line_potentials: core.Potential):
    """The chart of entropy and temperature, this many pixels high, the entropy up and the logarithm of the temperature
    across: the lines of the highest and the lowest field `[lines, samples] State-` with their potentials, and the ideal
    cycle through its corners `[corners] Scalar+`, all faint; and, empty, the line of the magnet's field and the dots of
    the crystal's states, for each frame to set through `chart_readings`. The figure, the line and the dots."""
    corner_s, corner_t = corner_entropies.to_array(), corner_temperatures.to_array()   # [corners], [corners]
    figure = plt.figure(figsize=(chart_width / charts.DPI, height / charts.DPI), dpi=charts.DPI)
    ax = figure.add_axes((0.2, 0.27, 0.75, 0.69))
    for states, salt_potentials in zip(line_states, line_potentials):
        ax.plot(*chart_readings(states, salt_potentials)[::-1], color=line_colour)
    ax.plot(np.append(corner_t, corner_t[0]), np.append(corner_s, corner_s[0]), color=ideal_colour)
    ax.set_xscale("log")
    ax.set_xticks(named_temperatures, temperature_names)
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_yticks([0, np.log(2)], ["0", "ln 2"])
    ax.set(xlim=chart_temperature_range, ylim=chart_entropy_range, xlabel="temperature", ylabel="entropy per spin")
    ax.spines[["top", "right"]].set_visible(False)
    field_line, = ax.plot([], [], color=magnet_colour, linewidth=charts.points(field_line_px))
    dots, = ax.plot([], [], "o", color=state_colour, markersize=charts.points(state_dot), markeredgecolor=state_rim,
                    markeredgewidth=charts.points(1.0))
    return figure, field_line, dots


def chart_readings(states: core.State, salt_potentials: core.Potential) -> tuple[np.ndarray, np.ndarray]:
    """The entropies and the temperatures `[...]` of these states with their potentials, read by pairing: the entropy
    by the plane along s, the temperature by a unit of entropy."""
    return (mv.s & states).to_array(), (salt_potentials & core.entropy).to_array()


# --- drawing --------------------------------------------------------------------------
@charts.style()
def draw_equation_of_state(states: core.State, salt_potentials: core.Potential, moment: float) -> Figure:
    """The magnetization against the field over the temperature, for these states and their potentials `[shares]`, and
    Curie's law, the magnetization the moment squared times the field over the temperature, as a dashed line up to
    saturation."""
    shares = (core.magnet_plane & states).to_array() / moment                 # [shares]
    ratios = ((salt_potentials & core.spin) / (salt_potentials & core.entropy)).to_array()   # [shares]
    figure, ax = charts.subplots(*state_size, layout="constrained")
    ax.plot(ratios, shares, color=profile_colour)
    ax.plot([0, 1 / moment], [0, 1], "--", color="0.55")
    ax.set(xlim=(0, ratios.max()), ylim=(0, 1.05), xlabel="field / temperature (T/K)", ylabel="magnetization")
    ax.set_xticks([0, 1, 2, 3])
    ax.set_yticks([0, 1], ["0", "saturation"])
    ax.spines[["top", "right"]].set_visible(False)
    return figure


@charts.style()
def draw_cycle(corner_entropies: core.Scalar, corner_temperatures: core.Scalar, line_states: core.State,
               line_potentials: core.Potential) -> Figure:
    """The ideal cycle through these corners `[corners] Scalar+` on the temperature-entropy chart, the logarithm of the
    temperature up and the entropy across, over the lines of equal field `[lines, samples] State-` with their
    potentials, each named by its field at its warmest end."""
    corner_s, corner_t = corner_entropies.to_array(), corner_temperatures.to_array()   # [corners], [corners]
    line_s, line_t = (mv.s & line_states).to_array(), (line_potentials & core.entropy).to_array()
    line_fields = (line_potentials[:, -1] & core.spin).to_array()             # [lines]
    figure, ax = charts.subplots(*cycle_size, layout="constrained")
    for entropies, temperatures, field in zip(line_s, line_t, line_fields):
        ax.plot(entropies, temperatures, color=line_colour, linewidth=charts.points(line_px))
        name = f"{field:g} T" if field >= 0.1 else f"{1000 * field:g} mT"
        ax.annotate(name, (entropies[-1], temperatures[-1]), textcoords="offset points",
                    xytext=name_offset, ha="right", va="center", color=name_colour)
    ax.plot(np.append(corner_s, corner_s[0]), np.append(corner_t, corner_t[0]), color=cycle_colour,
            linewidth=charts.points(cycle_px))
    ax.set_yscale("log")
    ax.set_xticks([np.log(2)], ["ln 2"])
    ax.set_yticks(named_temperatures, temperature_names)
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.set(xlim=entropy_range, ylim=temperature_range, xlabel="entropy per spin", ylabel="temperature")
    ax.spines[["top", "right"]].set_visible(False)
    return figure


def draw_crystal(topology: Topology, vertices: Extensor, sites: np.ndarray, bath_temperature: float,
                 load_temperature: float) -> Figure:
    """The crystal at rest at its vertices `Point-[vertices]`, its triangles' edges over its fill, the sites its spins
    are drawn at `[sites]` marked, and the bath's and the load's bars at its ends, open, tinted by these
    temperatures."""
    points = vertices.batch()                                                 # [vertices] Point-
    _, end_edges = ends(topology, points)
    camera = scene_camera()
    return image_figure(on_white(drawing.composited([
        contours.picture(points, topology, np.zeros(topology.n_vertices), np.full(3, 0.88), camera),
        simplices.strokes(points, topology, outline_colour, edge_width, camera),
        *bars(points, end_edges, np.zeros(2, dtype=bool), tinted(np.array([bath_temperature, load_temperature])),
              camera),
        simplices.markers(points[sites], spin_colour, 3.0, camera)])))


def animate_crystal(topology: Topology, vertices: Extensor, sites: np.ndarray, bath_temperature: float,
                    load_temperature: float, highest_field: float, corner_entropies: core.Scalar,
                    corner_temperatures: core.Scalar, line_states: core.State, line_potentials: core.Potential,
                    frames: Iterable[tuple[core.State, core.Potential, core.Magnetic, core.Scalar, core.Spin,
                                           core.State, core.Potential]]) -> Iterator[np.ndarray]:
    """The crystal at its vertices `Point-[vertices]` at each frame as it comes, its states `State-[vertices]`, its
    potentials `Potential+[vertices]`, the magnet's field `[] Magnetic+`, the links `Scalar+[vertices]`, a spin of each
    cell `Spin-[vertices]` and the salt's states at the magnet's field `[samples] State-` with their potentials: the
    scene as `scene` draws it, its spins at these sites `[sites]`, and beside it, as high, the chart of entropy and
    temperature as `chart` draws it, with the line of the magnet's field and, as dots on it, the states along the
    crystal's middle."""
    points = vertices.batch()                                                 # [vertices] Point-
    end_vertices, end_edges = ends(topology, points)
    reservoir_colours = tinted(np.array([bath_temperature, load_temperature]))   # [ends, 3]
    camera = scene_camera()
    up = (mv.y & points).to_array().ravel()                                   # [vertices]
    # The row of vertices nearest the crystal's middle, for the chart's dots.
    middle = np.flatnonzero(np.isclose(up, up[np.argmin(np.abs(up - (up.min() + up.max()) / 2))]))
    with charts.style():
        figure, field_line, dots = chart(camera.shape[0], corner_entropies, corner_temperatures, line_states,
                                         line_potentials)
        for states, salt_potentials, field, links, spins, field_states, field_potentials in frames:
            field_line.set_data(*chart_readings(field_states, field_potentials)[::-1])
            entropies, temperatures = chart_readings(states.batch(), salt_potentials.batch())
            dots.set_data(temperatures[middle], entropies[middle])
            yield np.concatenate([scene(topology, points, end_vertices, end_edges, sites, reservoir_colours,
                                        highest_field, salt_potentials, field, links, spins, camera),
                                  capture(figure)], axis=1)
        plt.close(figure)
