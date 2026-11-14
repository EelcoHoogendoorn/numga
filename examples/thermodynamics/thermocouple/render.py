"""Planar temperature tints, electrochemical potential contours, and transient charts display the two-metal thermocouple.

Camera planes `x` and `y` read spatial positions weighed by `w & point`. Triangle temperatures interpolate across corners to tint the hairpin from blue below equilibrium through neutral grey to warm red, overlaid with level contours of the electrochemical potential, gold contact feet, and the top junction line.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.figure import Figure

from numga import stack
from numga.render import Camera, contours, drawing, simplices
from numga.render.drawing import Draw
from numga.topology import Topology
from examples.util import charts
from examples.util.panels import image_figure
from examples.thermodynamics.thermocouple import core

mv = core.mv
temperature_colours = LinearSegmentedColormap.from_list(
    "temperature", ((0.0, "#2f6eb5"), (0.5, "#7d8794"), (0.75, "#e2913f"), (1.0, "#c0391d"))
)
rest_colour = np.array(to_rgb("#dde2e8"))
outline_colour, outline_width = np.array(to_rgb("#4b4f55")), 1.5
contact_colour, contact_width = np.array(to_rgb("#b8860b")), 3.0
junction_width = 1.5
level_colour, level_ink, level_width = np.array(to_rgb("#1d2228")), 0.55, 1.0
n_levels = 12
margin, gap = 0.08, 24
chart_size = (400, 250)
voltage_colour, cooling_colour, axis_colour = "#2f6f9f", "#c8553d", "#555555"


# --- plumbing -------------------------------------------------------------------------
def white_composite(picture: np.ndarray) -> np.ndarray:
    """Alpha blending composites an RGBA image (`[..., 4]`) onto a white background as 8-bit RGB bytes (`[..., 3]`)."""
    canvas = picture[..., :3] + (1 - picture[..., 3:])
    return np.round(np.clip(canvas, 0, 1) * 255).astype(np.uint8)


def potential_readings(potentials: core.Potential, quantity: core.Parcel) -> np.ndarray:
    """Regressive pairing of `potentials` (`Potential+[V]`) with `quantity` (`Parcel-`) extracts scalar vertex readings (`[V]`)."""
    return (potentials.batch() & quantity).to_array()


def hairpin_camera(vertices: core.Point, resolution: float) -> Camera:
    """Coordinate bounds of unit-weight `vertices` (`Point-[V]`) with `margin` padding define the orthographic planar `Camera` at `resolution` pixels per unit width."""
    coordinates = (stack((mv.x, mv.y)) & vertices.batch()[:, None]).to_array()   # [V, 2]
    lower_bound, upper_bound = coordinates.min(axis=0) - margin, coordinates.max(axis=0) + margin
    return Camera(
        mv.x,
        mv.y,
        0 * mv.x,
        mv.w,
        0.0,
        (lower_bound + upper_bound) / 2,
        (upper_bound - lower_bound) / 2,
        round((upper_bound - lower_bound)[0] * resolution),
    )


def outline(
    topology: Topology,
    vertices: core.Point,
    metals: np.ndarray,
    contacts: tuple[np.ndarray, np.ndarray],
    camera: Camera,
) -> list[Draw]:
    """Border edges, bimetallic junction edges, and contact feet form the three stroke layers over the hairpin."""
    points = vertices.batch()
    contact_edges = np.isin(topology.elements[1], np.concatenate(contacts)).all(axis=-1)               # [E]
    # An interior edge lies on the junction when one of its two incident triangles belongs to each metal.
    metal_indicators = np.stack([metals, np.ones_like(metals)])                                        # [2, F]
    second_count, incident_faces = topology.incidence[2, 1].adjugate()(metal_indicators[:, None]).sum(axis=1)   # [E]
    junction_edges = (second_count == 1) & (incident_faces == 2)                                       # [E]
    return [
        simplices.strokes(points, topology.border, outline_colour, outline_width, camera),
        simplices.strokes(points, topology.subset(junction_edges, 1), outline_colour, junction_width, camera),
        simplices.strokes(points, topology.subset(contact_edges, 1), contact_colour, contact_width, camera),
    ]


def layers(
    topology: Topology,
    vertices: core.Point,
    metals: np.ndarray,
    contacts: tuple[np.ndarray, np.ndarray],
    potentials: core.Potential,
    reach: float,
    spacing: float,
    camera: Camera,
) -> list[Draw]:
    """Temperature tint, electrochemical potential contours, and border strokes compose one hairpin panel from `potentials` (`Potential+[V]`)."""
    points = vertices.batch()
    temperatures = potential_readings(potentials, core.entropy)                # [V]
    lines = contours.Lines(spacing, spacing / 2, np.append(level_colour, level_ink), level_width)
    return [
        contours.picture(
            points,
            topology,
            temperatures,
            temperature_colours(np.clip(0.5 + temperatures / (2 * reach), 0, 1))[:, :3],
            camera,
        ),
        contours.picture(
            points,
            topology,
            potential_readings(potentials, core.time),
            np.zeros(3),
            camera,
            lines=(lines,),
            opacity=0.0,
        ),
        *outline(topology, vertices, metals, contacts, camera),
    ]


# --- drawing --------------------------------------------------------------------------
def hairpin_figure(
    topology: Topology,
    vertices: core.Point,
    metals: np.ndarray,
    contacts: tuple[np.ndarray, np.ndarray],
    resolution: float,
) -> Figure:
    """Uniform fill and boundary strokes depict the unperturbed hairpin with its top junction and gold contact feet."""
    V, E, F = topology.n_elements
    camera = hairpin_camera(vertices, resolution)
    fill = contours.picture(vertices.batch(), topology, np.zeros(V), rest_colour, camera)
    return image_figure(white_composite(drawing.composited([fill, *outline(topology, vertices, metals, contacts, camera)])))


def animation(
    topology: Topology,
    vertices: core.Point,
    metals: np.ndarray,
    contacts: tuple[np.ndarray, np.ndarray],
    resolution: float,
    scale_frame: tuple[core.Potential, core.Potential],
    frames: Iterable[tuple[core.Potential, core.Potential]],
) -> Iterator[np.ndarray]:
    """Yields side-by-side RGB frames comparing junction heating on the left against current-driven Peltier cooling on the right."""
    camera = hairpin_camera(vertices, resolution)
    scales = [
        (np.abs(potential_readings(state, core.entropy)).max(), np.ptp(potential_readings(state, core.time)) / n_levels)
        for state in scale_frame
    ]
    divider = np.full((camera.shape[0], gap, 3), 255, dtype=np.uint8)
    for states in frames:
        heated_panel, driven_panel = (
            white_composite(
                drawing.composited(layers(topology, vertices, metals, contacts, state, reach, spacing, camera))
            )
            for state, (reach, spacing) in zip(states, scales)
        )
        yield np.concatenate([heated_panel, divider, driven_panel], axis=1)


@charts.style()
def readings_figure(times: np.ndarray, voltage: core.Scalar, cooling: core.Scalar) -> Figure:
    """Time-series curves compare the Seebeck foot voltage per unit heating against the Peltier junction cooling per unit current (`[...] Scalar+`)."""
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
