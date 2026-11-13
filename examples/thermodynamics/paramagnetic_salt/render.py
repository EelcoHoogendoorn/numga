"""Drawing for adiabatic demagnetization in a crystal: the slab tinted by its temperature, the magnet's field as the gap
around it, a wash as deep as the field is strong and a rim turning the way the field turns, the bath and the load as
bars at its ends, touching it while linked, and its magnetization as patches on a coarse lattice of its sites; under it
the temperature along its middle, against the place along it; and the salt's equation of state and the ideal cycle on
the temperature-entropy chart as charts.

The field and the magnetization are patches of one plane of space, the slab's own. A spin's patch is a disc in that
plane, its area growing with the magnetization's share of saturation, with an arc around its rim ending in a chevron for
the sense its spins turn in: read by pairing the magnet's plane with the magnetization. The field fills the magnet's
gap, read by pairing it with a unit magnetization, `field & spin`, against the highest field. The crystal's places are
read by the planes x and y and weighed by w. Its tint runs evenly in the logarithm of the temperature between fixed
limits, deep blue through pale yellow to deep red, so that a decade reads as a gradient; the profile under it is read
off the same states, by `potentials & entropy`."""
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
# The crystal's tint, evenly in the logarithm of the temperature between these limits.
temperature_colours = colormaps["RdYlBu_r"]
coldest, warmest = 3e-4, 5.0
# The magnet's gap around the crystal, in the crystal's widths, its wash's colour and its depth at the highest field;
# its rim's width at the highest field, and its chevrons' strokes' length, in pixels.
gap_low, gap_high = np.array([-0.3, -0.22]), np.array([2.3, 1.22])
field_colour, wash_depth = np.array(to_rgb("dodgerblue")), 0.4
rim_width, rim_chevron = 4.0, 9.0
# The crystal's outline and its triangles' edges, their colour and widths, and the reservoirs' bars' width, their
# outline's width, in the outline's colour, and their gap from the crystal while open, in pixels.
outline_colour, outline_width, edge_width = np.array(to_rgb("#3d4148")), 1.5, 1.0
bar_width, bar_outline, bar_gap = 5.0, 1.5, 7.0
# The magnetization's patches: their colour, how much a filled disc covers, its radius at saturation in the crystal's
# widths; the rim's arc: how far outside the disc it runs, as a share of the disc's radius, where around the rim it
# starts and how far it runs, as angles, its samples, its width and its chevron's strokes' length in pixels, the
# chevron no longer than this share of the disc's radius.
patch_colour, patch_opacity, patch_radius = np.array(to_rgb("#2b2140")), 0.3, 0.12
arc_reach, arc_start, arc_span, n_arc = 1.25, np.radians(-60.0), np.radians(240.0), 24
arc_width, arc_chevron, chevron_share = 1.6, 6.0, 0.8
# The scene's span in the crystal's widths, across and up, and its pixels to a width; the profile's height in pixels,
# its line's colour, the reservoirs' lines' colour and the axes' colour, and the temperatures named on it.
scene_low, scene_high, pixels_per_width = np.array([-0.4, -0.3]), np.array([2.4, 1.3]), 150
profile_height, profile_colour, reservoir_colour, axis_colour = 130, "#1d2228", "0.6", "#555555"
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


def gap(share: float, washed: bool, camera: Camera) -> list:
    """The magnet's gap around the crystal at this share of the highest field: a wash as deep as the share where
    `washed`, and its rim, as wide as the share, each side ending in a chevron running the way the field turns."""
    corners = place(np.array([gap_low[0], gap_high[0], gap_high[0], gap_low[0]]),
                    np.array([gap_low[1], gap_low[1], gap_high[1], gap_high[1]]))   # [4] Point-
    width = rim_width * share
    rim = simplices.strokes(corners, Topology.loops([4]), field_colour, width, camera, chevrons=np.ones(4),
                            chevron=rim_chevron * np.sqrt(share))
    wash = contours.picture(corners, Topology(4, [[0, 1, 2], [0, 2, 3]]), np.zeros(4), field_colour, camera,
                            opacity=wash_depth * share)
    return [wash, rim] if washed else [rim]


def patches(middles: Extensor, shares: np.ndarray, filled: bool, camera: Camera) -> list:
    """The magnetization's patches at these middles `[sites] Point-`, at these shares of saturation `[sites]`, negative
    where they turn the other way: discs of the magnet's plane, filled where `filled`, their area growing with the
    share, each with an arc around its rim ending in a chevron, running the way its spins turn."""
    sizes = patch_radius * np.sqrt(np.abs(shares))                            # [sites]
    angles = np.sign(shares)[:, None] * (arc_start + np.linspace(0, arc_span, n_arc))   # [sites, arc]
    # Each rim's places, the direction across turned in the magnet's plane, the other way round where the patch turns
    # the other way.
    turns = (core.magnet_plane * (-angles / 2)).exp()                         # [sites, arc] Rotor+
    rims = middles[:, None] + (arc_reach * sizes[:, None]) * (turns >> mv.x.dual())   # [sites, arc] Point-
    ending = np.tile(np.arange(n_arc - 1) == n_arc - 2, len(shares))          # [sites * arc edges]
    chevrons = simplices.strokes(rims.reshape(-1), Topology.chains([n_arc] * len(shares)), patch_colour, arc_width,
                                 camera, chevrons=ending, chevron=min(arc_chevron, chevron_share * sizes.min() /
                                                                       camera.pixel))
    discs = simplices.markers(middles, patch_colour, sizes / camera.pixel, camera, opacity=patch_opacity)
    return [discs, chevrons] if filled else [chevrons]


def profile_figure(width: int, along: np.ndarray, bath_temperature: float, load_temperature: float):
    """The profile's figure, this many pixels wide, its axes spanning the scene's span across so that a place along the
    crystal `along` `[profile]` stands under the crystal's own; the logarithm of the temperature up, the reservoirs'
    temperatures as faint lines across the crystal, named beyond its far end; and its line, empty."""
    figure = plt.figure(figsize=(width / charts.DPI, profile_height / charts.DPI), dpi=charts.DPI)
    ax = figure.add_axes((0, 0.06, 1, 0.92))
    ax.set_xlim(scene_low[0], scene_high[0])
    ax.set_yscale("log")
    ax.set_ylim(coldest, warmest)
    for temperature, name in ((bath_temperature, "bath"), (load_temperature, "load")):
        ax.plot([along.min(), along.max()], [temperature] * 2, "--", color=reservoir_colour,
                linewidth=charts.FRAME_WIDTH)
        ax.annotate(name, (along.max(), temperature), textcoords="offset points", xytext=(6, 0), va="center",
                    color=reservoir_colour)
    # The temperature axis at the crystal's near end, its ticks named to its left, beyond the crystal.
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.spines["left"].set_position(("data", along.min()))
    ax.spines["left"].set_color(axis_colour)
    ax.set_xticks([])
    ax.set_yticks(named_temperatures, temperature_names)
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.patch.set_alpha(0)
    line, = ax.plot([], [], color=profile_colour, linewidth=charts.LINE_WIDTH)
    return figure, line


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
    """The crystal at rest at its vertices `Point-[vertices]`, its triangles' edges over its fill, the sites of the
    magnetization's patches `[sites]` marked, and the bath's and the load's bars at its ends, open, tinted by these
    temperatures."""
    points = vertices.batch()                                                 # [vertices] Point-
    _, end_edges = ends(topology, points)
    camera = scene_camera()
    return image_figure(on_white(drawing.composited([
        contours.picture(points, topology, np.zeros(topology.n_vertices), np.full(3, 0.88), camera),
        simplices.strokes(points, topology, outline_colour, edge_width, camera),
        *bars(points, end_edges, np.zeros(2, dtype=bool), tinted(np.array([bath_temperature, load_temperature])),
              camera),
        simplices.markers(points[sites], patch_colour, 3.0, camera)])))


def animate_crystal(topology: Topology, vertices: Extensor, sites: np.ndarray, bath_temperature: float,
                    load_temperature: float, highest_field: float, moment: float,
                    frames: Iterable[tuple[core.State, core.Potential, core.Magnetic, core.Scalar]],
                    washed: bool = True, filled: bool = False) -> Iterator[np.ndarray]:
    """The crystal at its vertices `Point-[vertices]` at each frame as it comes, its states `State-[vertices]`, its
    potentials `Potential+[vertices]`, the magnet's field `[] Magnetic+` and the links `Scalar+[vertices]`: the slab
    tinted by its temperatures, the magnet's gap around it, washed where `washed`, the reservoirs' bars at its ends, the
    bath's and the load's tinted by their temperatures and touching it while linked, and the magnetization's patches at
    its sites `[sites]`, filled where `filled`; and under it the temperature along its middle."""
    points = vertices.batch()                                                 # [vertices] Point-
    end_vertices, end_edges = ends(topology, points)
    reservoir_colours = tinted(np.array([bath_temperature, load_temperature]))   # [ends, 3]
    camera = scene_camera()
    unmarked = np.zeros(topology.n_vertices)
    # The middle row of the crystal, in order along it, for the profile.
    along, up = (mv.x & points).to_array().ravel(), (mv.y & points).to_array().ravel()   # [vertices], [vertices]
    middle = np.flatnonzero(np.isclose(up, (up.min() + up.max()) / 2))
    middle = middle[np.argsort(along[middle])]                                # [profile]
    with charts.style():
        figure, line = profile_figure(camera.pixels, along[middle], bath_temperature, load_temperature)
        for states, salt_potentials, field, links in frames:
            temperatures = (salt_potentials & core.entropy).batch().to_array().ravel()   # [vertices]
            shares = ((core.magnet_plane & states) / moment).batch().to_array().ravel()  # [vertices]
            linked = links.batch().to_array().ravel() > 0                     # [vertices]
            scene = on_white(drawing.composited([
                *gap(float(((field & core.spin) / highest_field).to_array()), washed, camera),
                contours.picture(points, topology, unmarked, tinted(temperatures), camera),
                simplices.strokes(points, topology.border, outline_colour, outline_width, camera),
                *bars(points, end_edges, np.array([linked[vertices].any() for vertices in end_vertices]),
                      reservoir_colours, camera),
                *patches(points[sites], shares[sites], filled, camera)]))
            line.set_data(along[middle], temperatures[middle])
            yield np.concatenate([scene, capture(figure)], axis=0)
        plt.close(figure)
