"""Drawing for galvanomagnetic and thermomagnetic effects: the bar and the ring with their contacts, the flux of each
kind of carrier turned by the field, the conductors in their steady states as the carriers' balance is swept, and the
bar's voltages against the balance as a chart.

The conductors' points are read by the planes x and y and weighed by `mv.s & point`, the algebra's own projective
weight. A conductor is tinted by its temperature, interpolated across the triangles by their corners' hats, on one
palette: grey in equilibrium, towards red where it is warmer and towards blue where it is cooler, red and blue reached
at the electric reach where it is driven by a current, and at its contacts' temperatures where it is driven by heat.
Over the tint run the level lines of the electrochemical potential, and each kind of carrier's drift, one velocity to a
triangle, as a texture of streaks feathered along it, the electrons' white and the holes' black, a streak as long as
the drift there is fast against the median drift in the first state of the current-driven conductor, its ink the
kind's share of the carriers against the larger share, so that the more numerous kind is inked in full. The contacts
are gold. The magnetic field, a plane of space, is a disc in that plane, beside the bar's right end and in the ring's
hole, with an arc around its rim ending in a chevron for the sense it turns in.

The fluxes are read as offsets from the origin through the planes that read charge, `mv.xt` and `mv.yt`, and through
those that read entropy, `mv.xs` and `mv.ys`; the electric field through the planes each axis sweeps through time,
`field & (axis & time)`.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import chain

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator

from numga import concatenate, stack
from numga.extensor import Extensor
from examples.util import charts
from examples.util.wgsl import contours, flow, image_figure, simplices
from examples.util.wgsl.camera import Camera, offset_panel
from examples.thermodynamics.magnetotransport import core

Topology = simplices.Topology
mv = core.mv
State = tuple[np.ndarray, core.Potential, core.Spatial, core.Potential, core.Spatial]
# The conductors' colours, from cooler than equilibrium through its grey to warmer, by orange to red, and the change of
# temperature at which they reach red and blue in a conductor driven by a current, the Ettingshausen effect's; their
# fill at rest, their outline's colour and width, and their contacts' colour and width, in pixels.
temperature_colours = LinearSegmentedColormap.from_list(
    "temperature", ((0.0, "#2f6eb5"), (0.5, "#7d8794"), (0.75, "#e2913f"), (1.0, "#c0391d")))
electric_reach = 0.5
rest_colour = np.array(to_rgb("#dde2e8"))
outline_colour, outline_width = np.array(to_rgb("#4b4f55")), 1.5
contact_colour, contact_width = np.array(to_rgb("#b8860b")), 3.0
# The electrochemical potential's level lines: their colour, opacity and width in pixels, and how many steps they split
# the voltage between the contacts into in the first state.
level_colour, level_ink, level_width = np.array(to_rgb("#1d2228")), 0.55, 1.0
n_levels = 6
# The streaks of the two drifts: the electrons' colour and the holes', a white and a near-black that stand against the
# temperature's colours and against each other; the picture's cells one seed is scattered in, in pixels, sparse enough
# that each streak stands on its own, a little different for the two kinds, so that their streaks start apart where
# they drift alike; a seed's speck's radius, in pixels, which is the streak's half width; a streak's length at the
# reference speed, in pixels, along the bar and in the ring's narrower band; and how much the ink of the more numerous
# kind covers.
streak_colours = np.array([to_rgb("#f8f8f8"), to_rgb("#15181c")])
streak_grains, streak_speck, streak_ink = np.array([20.0, 23.0]), 1.4, 1.0
bar_streak_length, ring_streak_length = 45.0, 40.0
# The magnetic field's disc: its colour and opacity; beside the bar, its radius in widths and its gap from the bar; in
# the ring's hole, the share of the hole's radius its arc leaves free; its rim's arc: how far outside the disc it runs,
# as a share of the disc's radius, where around the rim it starts, from the direction along the bar, and how far it
# runs, as angles, its samples, its width and its chevron's strokes' length, in pixels.
magnetic_colour, patch_opacity = np.array(to_rgb("dodgerblue")), 0.8
patch_reach, patch_gap = 0.22, 0.2
hole_margin = 0.15
arc_reach = 1.25
arc_start, arc_span = np.radians(-60.0), np.radians(240.0)
n_arc = 32
arc_width, arc_chevron = 1.8, 8.0
# The pictures' room around a conductor, in its units, and the white gap between pictures set together, in pixels.
margin, gap = 0.08, 8
# The turns' diagram: each panel's pixels and the readings it spans across and up, the field's colour, the arrows'
# widths and their chevrons' strokes' length in pixels, the electrons' arrow's colour, darker than their streaks so
# that it shows on white, the flux at compensation's colour, and the dots of the path the flux runs along as the
# balance is swept.
turn_size, turn_across, turn_up = (220, 150), (-0.35, 1.1), (-0.5, 0.5)
electric_colour = np.array(to_rgb("crimson"))
arrow_width, total_width, arrow_chevron = 2.0, 3.0, 9.0
electron_arrow_colour, total_colour = np.array(to_rgb("#9aa3ad")), np.array(to_rgb("#6d28d9"))
sweep_dots = (1.5, 3.0)
# The readings' chart, three panels stacked over one axis of the balance: its pixels, the curves' colours, and the
# balances named on its axis.
chart_size = (430, 460)
along_colour, across_colour, heat_colour = "#4f5966", "#c2410c", "#1d4ed8"
balance_ticks, balance_names = [-1, 0, 1], ["electrons", "compensated", "holes"]


# --- plumbing -------------------------------------------------------------------------
def on_white(picture: np.ndarray) -> np.ndarray:
    """Colour and coverage `[rows, columns, 4]` laid over white, as bytes `[rows, columns, 3]`."""
    shown = picture[..., :3] + (1 - picture[..., 3:])
    return np.round(np.clip(shown, 0, 1) * 255).astype(np.uint8)


def readings(plane: Extensor, values: Extensor) -> np.ndarray:
    """Values read by a plane, as numbers in their batch and field's shape."""
    return (plane & values.batch()).to_array().reshape(values.batch().shape)


def potential_of(potentials: Extensor, amount: core.Parcel) -> np.ndarray:
    """The potential of a unit of this amount, the electrochemical potential of a charge or the temperature of an
    entropy, read off the potentials `[...] Potential+[vertices]`, as numbers in their batch and field's shape."""
    return (potentials.batch() & amount).to_array().reshape(potentials.batch().shape)


def tinted(temperatures: np.ndarray, reach: float) -> np.ndarray:
    """The colours `[vertices, 3]` of the temperatures `[vertices]` on the one palette: grey in equilibrium, red this
    reach above it and blue this reach below."""
    return temperature_colours(np.clip(0.5 + temperatures / (2 * reach), 0, 1))[:, :3]


def framing(vertices: core.Point, beside: float, resolution: float) -> Camera:
    """The camera looking down on a conductor and this much room beside it, past its reach along the picture, with the
    margin around them, at this resolution in pixels to the conductor's unit: points read by the planes x and y,
    weighed by their weight along s."""
    along, across = readings(mv.x, vertices), readings(mv.y, vertices)
    low = np.array([along.min(), across.min()]) - margin
    high = np.array([along.max() + beside, across.max()]) + margin
    return Camera(mv.x, mv.y, 0 * mv.x, mv.s, 0.0, (low + high) / 2, (high - low) / 2,
                  round((high - low)[0] * resolution))


def outline(topology: Topology, vertices: core.Point, contacts: tuple[np.ndarray, np.ndarray], camera: Camera) -> list:
    """A conductor's outline, and over it its contacts `[contact]` each, in gold."""
    points = vertices.batch()
    on_contacts = np.isin(topology.elements[1], np.concatenate(contacts)).all(axis=-1)   # [edges]
    return [simplices.strokes(points, topology.border, outline_colour, outline_width, camera),
            simplices.strokes(points, topology.subset(on_contacts, 1), contact_colour, contact_width, camera)]


def field_patch(centre: core.Point, magnetic: core.Magnetic, radius: float, camera: Camera) -> list:
    """The magnetic field `[] Bivector+`, a plane of space: a disc about the centre in that plane, this radius, and an
    arc around its rim ending in a chevron, running the way the field turns."""
    angles = arc_start + np.linspace(0, arc_span, n_arc)                      # [arc]
    # The rim's places, the direction along the bar turned in the field's own plane, so the arc runs the field's way.
    turns = (magnetic.normalized() * (-angles / 2)).exp()                     # [arc] Rotor+
    arc = centre + arc_reach * radius * (turns >> core.axis)                  # [arc] Point-
    ending = np.arange(n_arc - 1) == n_arc - 2                                # [arc edges]
    return [simplices.markers(stack([centre]), magnetic_colour, radius / camera.pixel, camera, opacity=patch_opacity),
            simplices.strokes(arc, Topology.chains([n_arc]), magnetic_colour, arc_width, camera, chevrons=ending,
                              chevron=arc_chevron)]


def face_corners(topology: Topology, vertices: core.Point) -> tuple[core.Point, Topology]:
    """Each triangle with its own three corners, for a field with one value to a triangle: the corners, corner after
    corner of every triangle, `[corners * faces] Point-`, and the triangles among them."""
    n_faces = topology.n_elements[2]
    corners = (topology.incidence[2, 0] * vertices).batch().reshape(-1)      # [corners * faces] Point-
    return corners, Topology(3 * n_faces, np.arange(3 * n_faces).reshape(3, n_faces).T)


def speeds(velocities: core.Spatial) -> np.ndarray:
    """The speed of each kind of carrier in each triangle `[kinds, faces]`, from its velocity `[kinds] Spatial-[faces]`
    read along and across."""
    return np.hypot(readings(mv.x, velocities), readings(mv.y, velocities))


def voltage(potentials: core.Potential, contacts: tuple[np.ndarray, np.ndarray]) -> float:
    """The voltage between a conductor's contacts `Potential+[vertices]`, the first less the second."""
    values = potential_of(potentials, core.time)
    return values[contacts[0][0]] - values[contacts[1][0]]


def layers(topology: Topology, vertices: core.Point, contacts: tuple[np.ndarray, np.ndarray],
           potentials: core.Potential, velocities: core.Spatial, shares: np.ndarray, reach: float, spacing: float,
           reference_speed: float, streak_length: float, camera: Camera) -> list:
    """A conductor's picture as layers: tinted by its temperature, red and blue this reach either way, the level lines
    of its electrochemical potential `spacing` apart, halfway between steps from its value in equilibrium, so that a
    potential left there draws none, from its potentials `Potential+[vertices]`; each kind of carrier's drift in each
    triangle `[kinds] Spatial-[faces]` streaked over it, inked by the kind's share of the carriers against the larger
    share `[kinds]`, a streak `streak_length` pixels long at the reference speed; and its outline and contacts."""
    points = vertices.batch()
    face_places, split = face_corners(topology, vertices)
    temperatures = potential_of(potentials, core.entropy)                     # [vertices]
    lines = contours.Lines(spacing, spacing / 2, np.append(level_colour, level_ink), level_width)
    # Each kind's velocity at every corner of its triangle, for the streaks' shader.
    corner_velocities = [concatenate([velocities[kind].batch()] * 3) for kind in range(len(streak_colours))]
    return [contours.picture(points, topology, temperatures, tinted(temperatures, reach), camera),
            contours.picture(points, topology, potential_of(potentials, core.time), np.zeros(3), camera,
                             lines=(lines,), opacity=0.0),
            *[flow.layer(face_places, split, velocity, camera, reference_speed, colour, length=streak_length,
                         grain_pixels=grain, speck_pixels=streak_speck, ink_opacity=streak_ink * share)
              for velocity, colour, grain, share in zip(corner_velocities, streak_colours, streak_grains, shares)],
            *outline(topology, vertices, contacts, camera)]


def drawn(topology: Topology, vertices: core.Point, contacts: tuple[np.ndarray, np.ndarray], patch: list,
          streak_length: float, camera: Camera, states: Iterable[State]) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """A conductor at each state as it comes, driven by a current and by heat, each as `layers` draws it with the
    field's patch: the current-driven one's tint reaching red and blue at the electric reach and the heat-driven one's
    at its contacts' temperatures, its level lines splitting each drive's voltage between the contacts in the first
    state into equal steps, and its streaks measured against the median drift in the first state driven by the
    current."""
    states = iter(states)
    first = next(states)
    _, first_electric, first_drifts, first_thermal, _ = first
    reaches = electric_reach, abs(potential_of(first_thermal, core.entropy)[contacts[0][0]])
    spacings = [abs(voltage(potentials, contacts)) / n_levels for potentials in (first_electric, first_thermal)]
    reference_speed = np.median(speeds(first_drifts))
    for densities, electric, electric_drifts, thermal, thermal_drifts in chain([first], states):
        shares = densities / densities.max()
        yield tuple(on_white(simplices.composited(
            layers(topology, vertices, contacts, potentials, velocities, shares, reach, spacing, reference_speed,
                   streak_length, camera) + patch))
            for potentials, velocities, reach, spacing in zip((electric, thermal), (electric_drifts, thermal_drifts),
                                                              reaches, spacings))


# --- drawing --------------------------------------------------------------------------
def draw_conductors(bar_topology: Topology, bar_vertices: core.Point, bar_contacts: tuple[np.ndarray, np.ndarray],
                    ring_topology: Topology, ring_vertices: core.Point, ring_contacts: tuple[np.ndarray, np.ndarray],
                    bar_resolution: float, ring_resolution: float) -> Figure:
    """The bar and the ring side by side, each a pale fill within its outline, its contacts in gold, at these
    resolutions in pixels to their units."""
    pictures = []
    for topology, vertices, contacts, resolution in ((bar_topology, bar_vertices, bar_contacts, bar_resolution),
                                                     (ring_topology, ring_vertices, ring_contacts, ring_resolution)):
        camera = framing(vertices, 0.0, resolution)
        fill = contours.picture(vertices.batch(), topology, np.zeros(topology.n_vertices), rest_colour, camera)
        pictures.append(on_white(simplices.composited([fill, *outline(topology, vertices, contacts, camera)])))
    # The bar set level with the ring's middle.
    height = pictures[1].shape[0]
    above = (height - pictures[0].shape[0]) // 2
    bar = np.full((height, pictures[0].shape[1], 3), 255, dtype=np.uint8)
    bar[above:above + pictures[0].shape[0]] = pictures[0]
    return image_figure(np.concatenate([bar, np.full((height, gap, 3), 255, np.uint8), pictures[1]], axis=1))


def animate_bars(topology: Topology, vertices: core.Point, contacts: tuple[np.ndarray, np.ndarray],
                 magnetic: core.Magnetic, resolution: float, states: Iterable[State]) -> Iterator[np.ndarray]:
    """The bar at each state as it comes, driven by a current above and by heat below, as `drawn` draws them, with the
    magnetic field `[] Bivector+` beside each bar's right end, at this resolution in pixels to a width."""
    camera = framing(vertices, patch_gap + 2 * arc_reach * patch_reach, resolution)
    # The patch's centre, a place of unit weight beside the bar's right end, on its middle line.
    beside = readings(mv.x, vertices).max() + patch_gap + arc_reach * patch_reach
    patch = field_patch(core.entropy + beside * core.axis, magnetic, patch_reach, camera)
    spacer = np.full((gap, camera.shape[1], 3), 255, dtype=np.uint8)
    for electric, thermal in drawn(topology, vertices, contacts, patch, bar_streak_length, camera, states):
        yield np.concatenate([electric, spacer, thermal], axis=0)


def animate_rings(topology: Topology, vertices: core.Point, contacts: tuple[np.ndarray, np.ndarray],
                  magnetic: core.Magnetic, resolution: float, states: Iterable[State]) -> Iterator[np.ndarray]:
    """The ring at each state as it comes, driven by a current on the left and by heat on the right, as `drawn` draws
    them, with the magnetic field `[] Bivector+` in each ring's hole, its arc reaching the hole's radius less the
    margin, at this resolution in pixels to an outer radius."""
    camera = framing(vertices, 0.0, resolution)
    # The hole's radius, the inner rim's nearest vertex's distance from the ring's centre, the origin.
    rim = topology.reading(contacts[0], 0) * vertices                            # Point-[inner]
    hole_radius = np.hypot(readings(mv.x, rim), readings(mv.y, rim)).min()
    patch = field_patch(core.entropy, magnetic, hole_radius * (1 - hole_margin) / arc_reach, camera)
    spacer = np.full((camera.shape[0], gap, 3), 255, dtype=np.uint8)
    for electric, thermal in drawn(topology, vertices, contacts, patch, ring_streak_length, camera, states):
        yield np.concatenate([electric, spacer, thermal], axis=1)


@charts.style()
def draw_turns(drive: core.Drive, fluxes: core.Flux) -> Figure:
    """The flux a unit electric field `[] Drive+` drives with electrons only and with holes only `[ends] Flux-`, read
    through the planes that read charge, on the left, and through those that read entropy, on the right: in each, the
    field as an arrow, the two fluxes as arrows, the electrons' grey and the holes' black, the path between their tips
    that the flux runs along as the balance is swept, dotted, and the flux at compensation, halfway along it."""
    figure, axes = charts.subplots(2 * turn_size[0], turn_size[1], ncols=2)
    for ax in axes:
        ax.set_xlim(*turn_across)
        ax.set_ylim(*turn_up)
        ax.set_aspect("equal")
        ax.set_axis_off()
    figure.tight_layout(pad=0.2)
    electrons, holes = fluxes
    compensated = (electrons + holes) / 2                                     # [] Flux-
    arrows = stack((0 * fluxes, fluxes), axis=-1).reshape(-1)                # [ends * 2] Flux-
    for ax, planes in zip(axes, ((mv.xt, mv.yt), (mv.xs, mv.ys))):
        field_view = offset_panel(ax, core.axis & core.time, core.across & core.time)
        flux_view = offset_panel(ax, *planes)
        simplices.overlay(ax, [
            simplices.strokes(stack((0 * drive, drive)), Topology.chains([2]), electric_colour, arrow_width,
                              field_view, chevrons=np.ones(1), chevron=arrow_chevron),
            simplices.strokes(fluxes, Topology.chains([2]), np.zeros(3), charts.LINE_PX, flux_view, dashes=sweep_dots,
                              cap=simplices.Cap.ROUND, opacity=0.5),
            simplices.strokes(arrows, Topology.chains([2, 2]), np.stack([electron_arrow_colour, streak_colours[1]]),
                              arrow_width, flux_view, chevrons=np.ones(2), chevron=arrow_chevron),
            simplices.strokes(stack((0 * compensated, compensated)), Topology.chains([2]), total_colour, total_width,
                              flux_view, chevrons=np.ones(1), chevron=arrow_chevron)])
    return figure


@charts.style()
def draw_readings(balances: np.ndarray, resistance: core.Scalar, hall: core.Scalar, seebeck: core.Scalar,
                  nernst: core.Scalar, righi_leduc: core.Scalar) -> Figure:
    """The bar's readings against the balance of its carriers `[frames] Scalar+` each, in panels stacked over one axis
    of the balance: driven by a current, its resistance along it and its Hall resistance, at the top; driven by heat,
    its Seebeck and its Nernst signal, in the middle, and its Righi-Leduc signal, at the bottom; each curve named at
    its end."""
    figure, axes = charts.subplots(*chart_size, nrows=3, sharex=True)
    for ax, curves, label, spacing in zip(axes, (((resistance, along_colour, "along"), (hall, across_colour, "Hall")),
                                                 ((seebeck, along_colour, "Seebeck"), (nernst, across_colour, "Nernst")),
                                                 ((righi_leduc, heat_colour, "Righi–Leduc"),)),
                                          ("resistance", "thermopower", "temperature\nacross"), (1, 1, 0.1)):
        for values, colour, name in curves:
            shown = values.to_array().reshape(-1)
            ax.plot(balances, shown, color=colour)
            ax.annotate(name, (balances[-1], shown[-1]), xytext=(4, 0), textcoords="offset points", va="center",
                        color=colour)
        ax.axhline(0.0, color="#555555", linewidth=charts.FRAME_WIDTH)
        ax.spines[["top", "right", "bottom"]].set_visible(False)
        ax.set(xlim=(balances[0], balances[-1]), ylabel=label)
        ax.yaxis.set_major_locator(MultipleLocator(spacing))
    axes[-1].set(xticks=balance_ticks, xticklabels=balance_names)
    figure.tight_layout(pad=0.4)
    return figure
