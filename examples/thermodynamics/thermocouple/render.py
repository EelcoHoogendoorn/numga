"""Drawing for the thermocouple: the hairpin heated at its junction and the hairpin driven by a current, side by side,
as they settle, the hairpin at rest, and the two readings against time as a chart.

The hairpin's points are read by the planes x and y and weighed by `mv.w & point`, the algebra's own projective
weight. A hairpin is tinted by its temperature, interpolated across the triangles by their corners' simplex gradients,
on one palette: grey in equilibrium, towards red where it is warmer and towards blue where it is cooler. Over the tint
run the level lines of the electrochemical potential. The feet are gold, and a line across the top marks the junction
between the metals.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.figure import Figure

from numga.extensor import Extensor
from numga.render import Camera, contours, drawing, simplices
from numga.topology import Topology
from examples.util import charts
from examples.util.panels import image_figure
from examples.thermodynamics.thermocouple import core

mv = core.mv
# The hairpin's colours, from cooler than equilibrium through its grey to warmer, by orange to red; its outline's
# colour and width, its feet's colour and width and its junction's width, in pixels.
temperature_colours = LinearSegmentedColormap.from_list(
    "temperature", ((0.0, "#2f6eb5"), (0.5, "#7d8794"), (0.75, "#e2913f"), (1.0, "#c0391d")))
rest_colour = np.array(to_rgb("#dde2e8"))
outline_colour, outline_width = np.array(to_rgb("#4b4f55")), 1.5
contact_colour, contact_width = np.array(to_rgb("#b8860b")), 3.0
junction_width = 1.5
# The electrochemical potential's level lines: their colour, opacity and width in pixels, and how many steps they split
# the potential's span into in the scale state.
level_colour, level_ink, level_width = np.array(to_rgb("#1d2228")), 0.55, 1.0
n_levels = 12
# The pictures' room around a hairpin, in its widths, and the white gap between the two, in pixels.
margin, gap = 0.08, 24
# The readings' chart: its pixels and its curves' colours.
chart_size = (400, 250)
voltage_colour, cooling_colour, axis_colour = "#2f6f9f", "#c8553d", "#555555"


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
    entropy, read off the potentials `Potential+[vertices]`, as numbers `[vertices]`."""
    return (potentials.batch() & amount).to_array().reshape(potentials.batch().shape)


def framing(vertices: core.Point, resolution: float) -> Camera:
    """The camera looking down on the hairpin, its points read across the picture by the plane x and up it by the
    plane y and weighed by their weight along w, with the margin around it, at this resolution in pixels to a
    width."""
    shown = np.stack([readings(mv.x, vertices), readings(mv.y, vertices)], axis=-1)   # [vertices, 2]
    low, high = shown.min(axis=0) - margin, shown.max(axis=0) + margin
    return Camera(mv.x, mv.y, 0 * mv.x, mv.w, 0.0, (low + high) / 2, (high - low) / 2,
                  round((high - low)[0] * resolution))


def outline(topology: Topology, vertices: core.Point, metals: np.ndarray, contacts: tuple[np.ndarray, np.ndarray],
            camera: Camera) -> list:
    """The hairpin's outline, its junction, the edges whose two triangles are of different metals, and over them its
    feet `[foot]` each, in gold."""
    points = vertices.batch()
    on_contacts = np.isin(topology.elements[1], np.concatenate(contacts)).all(axis=-1)   # [edges]
    # An edge lies on the junction where it bounds a triangle of each metal: of its two triangles, one is of the
    # second metal.
    second = np.stack([metals, np.ones_like(metals)])                          # [2, faces]
    of_second, bounded = topology.incidence[2, 1].adjugate()(second[:, None]).sum(axis=1)   # [edges] each
    on_junction = (of_second == 1) & (bounded == 2)                           # [edges]
    return [simplices.strokes(points, topology.border, outline_colour, outline_width, camera),
            simplices.strokes(points, topology.subset(on_junction, 1), outline_colour, junction_width, camera),
            simplices.strokes(points, topology.subset(on_contacts, 1), contact_colour, contact_width, camera)]


def layers(topology: Topology, vertices: core.Point, metals: np.ndarray, contacts: tuple[np.ndarray, np.ndarray],
           potentials: core.Potential, reach: float, spacing: float, camera: Camera) -> list:
    """The hairpin's picture as layers: tinted by its temperature, red and blue this reach either way, the level lines
    of its electrochemical potential `spacing` apart, halfway between steps from its value in equilibrium, so that a
    potential left there draws none, from its potentials `Potential+[vertices]`; and its outline, junction and
    feet."""
    points = vertices.batch()
    temperatures = potential_of(potentials, core.entropy)                     # [vertices]
    lines = contours.Lines(spacing, spacing / 2, np.append(level_colour, level_ink), level_width)
    return [contours.picture(points, topology, temperatures,
                             temperature_colours(np.clip(0.5 + temperatures / (2 * reach), 0, 1))[:, :3], camera),
            contours.picture(points, topology, potential_of(potentials, core.time), np.zeros(3), camera,
                             lines=(lines,), opacity=0.0),
            *outline(topology, vertices, metals, contacts, camera)]


# --- drawing --------------------------------------------------------------------------
def draw_hairpin(topology: Topology, vertices: core.Point, metals: np.ndarray, contacts: tuple[np.ndarray, np.ndarray],
                 resolution: float) -> Figure:
    """The hairpin at rest, a pale fill within its outline, the junction a line across its top and its feet in gold,
    at this resolution in pixels to a width."""
    camera = framing(vertices, resolution)
    fill = contours.picture(vertices.batch(), topology, np.zeros(topology.n_vertices), rest_colour, camera)
    return image_figure(on_white(drawing.composited([fill, *outline(topology, vertices, metals, contacts, camera)])))


def animate(topology: Topology, vertices: core.Point, metals: np.ndarray, contacts: tuple[np.ndarray, np.ndarray],
            resolution: float, scale_frame: tuple[core.Potential, core.Potential],
            frames: Iterable[tuple[core.Potential, core.Potential]]) -> Iterator[np.ndarray]:
    """Frames of the hairpin heated at its junction, on the left, and driven by a current, on the right, as `layers`
    draws them: each on the scales of its state in `scale_frame`, its tint reaching red or blue at that state's
    largest change of temperature, and its level lines splitting that state's span of the potential evenly."""
    camera = framing(vertices, resolution)
    scales = [(np.abs(potential_of(state, core.entropy)).max(), np.ptp(potential_of(state, core.time)) / n_levels)
              for state in scale_frame]
    blank = np.full((camera.shape[0], gap, 3), 255, dtype=np.uint8)
    for states in frames:
        heated, driven = (on_white(drawing.composited(
            layers(topology, vertices, metals, contacts, state, reach, spacing, camera)))
            for state, (reach, spacing) in zip(states, scales))
        yield np.concatenate([heated, blank, driven], axis=1)


@charts.style()
def draw_readings(times: np.ndarray, voltage: core.Scalar, cooling: core.Scalar) -> Figure:
    """The hairpin's two readings `[frames] Scalar+` each against time: heated, the voltage between its feet per unit
    of heating, a line; driven by a current, the fall of the junction's temperature per unit of current, dots on
    it."""
    figure, ax = charts.subplots(*chart_size)
    ax.plot(times, voltage.to_array(), color=voltage_colour)
    ax.plot(times[::5], cooling.to_array()[::5], "o", color=cooling_colour)
    ax.axhline(0.0, color=axis_colour, linewidth=charts.FRAME_WIDTH)
    ax.text(0.97, 0.3, "voltage per heating", color=voltage_colour, transform=ax.transAxes, ha="right")
    ax.text(0.97, 0.18, "cooling per current", color=cooling_colour, transform=ax.transAxes, ha="right")
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.set(xlim=(times[0], times[-1] * 1.02), xticks=[], yticks=[0], xlabel="time")
    figure.tight_layout(pad=0.4)
    return figure
