"""Rendering routines project periodic triangulated channels, flow streaks, equipotentials, dye bands, and generator efficiency charts.

Opening a closed channel strip along its periodic seam appends a shifted copy of the first vertex column one period downstream. Cameras pair vertex points against the coordinate planes `x` and `y` and weigh them by the projective plane `w`. Channels display ionic charge density or entropy production across triangles, flow streaks scaled by speed, equipotential contours, and wall strokes coloured magenta for positive surface charge and green for negative surface charge.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import to_rgb

from numga import concatenate
from numga.extensor import Extensor
from numga.render import Camera, contours, drawing, flow, simplices
from numga.topology import Topology
from examples.util.charts import style, subplots
from examples.util.panels import image_figure
from examples.fluids.electrokinetics import core

mv = core.mv
wall_colour = np.array(to_rgb("#3b4a57"))
liquid_colour = np.array(to_rgb("#eef1f4"))
positive_colour, negative_colour = np.array(to_rgb("#c51b7d")), np.array(to_rgb("#4d9221"))
streak_colour = np.array(to_rgb("#2b2f33"))
streak_ink = 0.6
neutral_colour = np.array(to_rgb("#1f4e79"))
pumped_colour = neutral_colour
dye_margin, dye_full = 0.06, 0.4
glow_strength = 0.5
n_equipotentials = 16
equipotential_colour, equipotential_ink, equipotential_width = np.array(to_rgb("#3b4a57")), 0.6, 1.5
efficiency_map, heat_map, charge_map = (colormaps[name] for name in ("Blues", "Oranges", "PiYG_r"))
efficiency_shades = (0.03, 0.6)
efficiency_levels = np.array([0.01, 0.02, 0.04, 0.06])
efficiency_width, efficiency_height = 320, 260
profile_width, profile_height = 300, 220
profile_colours = (np.array(to_rgb("#b35806")), neutral_colour)
profile_labels = ("pressure drop", "field")
profile_speeds = (0, 1, 1.5)
n_pixels, margin, gap_pixels = 300, 0.4, 8
wall_width = 3.0
field_strength = 0.85


# --- plumbing -------------------------------------------------------------------------
def white_composite(picture: np.ndarray) -> np.ndarray:
    """Premultiplied RGBA `picture` (`[n_rows, n_columns, 4]`) composites over a white background as RGB bytes."""
    shown = picture[..., :3] + (1 - picture[..., 3:])
    return np.round(np.clip(shown, 0, 1) * 255).astype(np.uint8)


def plane_readings(plane: Extensor, values: Extensor) -> np.ndarray:
    """Pairing `plane` against `values` extracts coordinate numbers in their batch and site shape."""
    batched = values.batch()
    return (plane & batched).to_array().reshape(batched.shape)


def opened_strip(topology: Topology, vertices: Extensor, n_across: int, period: float) -> tuple[Extensor, Topology]:
    """Duplicating the first column of `n_across` vertices one `period` downstream opens a periodic strip `topology` at its seam."""
    V, E, F = topology.n_elements
    faces = topology.elements[2]
    columns = faces // n_across
    across_seam = (columns.max(axis=-1) - columns.min(axis=-1) > 1)[:, None] & (columns == 0)
    opened_faces = np.where(across_seam, V + faces % n_across, faces)
    points = vertices.batch()                                                 # [V] Point-
    first_column = points[:n_across] + core.axis * period                     # [n_across] Point-
    return concatenate([points, first_column]), Topology(V + n_across, opened_faces)


def strip_camera(vertices: Extensor, period: float, pixels: int) -> Camera:
    """An orthographic `Camera` of width `pixels` frames `vertices` over `period` using projective weight `w`."""
    widest = np.abs(plane_readings(mv.y, vertices)).max()
    return Camera(
        mv.x, mv.y, 0 * mv.x, mv.w, 0.0,
        np.array([period / 2, 0.0]), np.array([period / 2, widest + margin]), pixels,
    )


def wall_strokes(points: Extensor, n_across: int, wall_charges: np.ndarray, camera: Camera) -> np.ndarray:
    """Lower and upper boundary polylines of `points` render in magenta where `wall_charges` is positive and green where negative."""
    rows = points.reshape(-1, n_across)
    closed_charges = np.append(wall_charges, wall_charges[:1])
    edge_colours = np.where(((closed_charges[:-1] + closed_charges[1:]) / 2)[:, None] > 0, positive_colour, negative_colour)
    n_stations = rows.shape[0]
    return simplices.strokes(
        concatenate([rows[:, 0], rows[:, -1]]),
        Topology.chains([n_stations, n_stations]),
        np.concatenate([edge_colours, edge_colours]),
        wall_width,
        camera,
    )


def field_layer(points: Extensor, opened: Topology, values: np.ndarray, colours: np.ndarray, camera: Camera) -> np.ndarray:
    """Vertex `colours` interpolate across the triangles of `opened` under `camera`."""
    return contours.picture(points, opened, values, colours, camera)


def equipotential_lines(
    points: Extensor,
    opened: Topology,
    potentials: Extensor,
    falls: Extensor,
    n_across: int,
    camera: Camera,
) -> list[np.ndarray]:
    """Uniform contour levels across `potentials` (`[...] Scalar+[V]`) and periodic `falls` render one equipotential overlay per strip."""
    V_opened, E_opened, F_opened = opened.n_elements
    drops = falls.to_array().reshape(-1)
    n_strips = len(drops)
    values = potentials.to_array().reshape(n_strips, -1)
    levels = np.concatenate([values, values[:, :n_across] - drops[:, None]], axis=-1)  # [n_strips, V_opened]
    spacing = np.abs(drops).max() / n_equipotentials
    line_spec = contours.Lines(spacing, spacing / 2, np.append(equipotential_colour, equipotential_ink), equipotential_width)
    return [
        contours.picture(points, opened, level, np.zeros((V_opened, 3)), camera, lines=(line_spec,), opacity=0.0)
        for level in levels
    ]


def strip_stack(images: list[np.ndarray]):
    """Vertical concatenation with white `gap_pixels` separators displays `images` in one figure."""
    gap = np.full((gap_pixels, images[0].shape[1], 3), 255, dtype=np.uint8)
    return image_figure(np.concatenate([part for image in images for part in (image, gap)][:-1]))


def seam_values(values: np.ndarray, n_across: int) -> np.ndarray:
    """Appending the first `n_across` vertex entries closes `values` across the opened periodic seam."""
    return np.concatenate([values, values[..., :n_across]], axis=-1)


def channels_figure(topology: Topology, vertices: Extensor, n_across: int, period: float, wall_charges: np.ndarray):
    """Stacked channel panels display the pale liquid interior bounded by wall strokes coloured by `wall_charges`."""
    points, opened = opened_strip(topology, vertices, n_across, period)
    V_opened, E_opened, F_opened = opened.n_elements
    camera = strip_camera(vertices, period, n_pixels)
    liquid = field_layer(points, opened, np.zeros(V_opened), np.tile(liquid_colour, (V_opened, 1)), camera)
    return strip_stack([
        white_composite(drawing.composited([liquid, wall_strokes(points, n_across, charges, camera)]))
        for charges in wall_charges
    ])


def liquid_layers(
    points: Extensor,
    opened: Topology,
    n_across: int,
    charge: Extensor,
    speeds: Extensor,
    potentials: Extensor,
    falls: Extensor,
    camera: Camera,
) -> list[list[np.ndarray]]:
    """Charge-density shading, velocity streaks, and equipotential contours form the three RGBA layers for each strip."""
    V_opened, E_opened, F_opened = opened.n_elements
    densities = seam_values(charge.batch().to_array(), n_across)
    largest = np.abs(densities).max()
    fastest = np.abs(plane_readings(mv.x, speeds)).max()
    velocities = speeds.batch()                                               # [..., V] Point-
    closed_velocities = concatenate([velocities, velocities[:, :n_across]], axis=-1)  # [..., V_opened] Point-
    return [
        [
            field_layer(points, opened, density, 1 - field_strength * (1 - charge_map(0.5 + density / (2 * largest))[:, :3]), camera),
            flow.layer(points, opened, velocity, camera, fastest, streak_colour) * np.array([1, 1, 1, streak_ink]),
            lines,
        ]
        for density, velocity, lines in zip(
            densities, closed_velocities, equipotential_lines(points, opened, potentials, falls, n_across, camera))
    ]


def flow_figure(
    topology: Topology,
    vertices: Extensor,
    n_across: int,
    period: float,
    charge: Extensor,
    speeds: Extensor,
    potentials: Extensor,
    falls: Extensor,
    wall_charges: np.ndarray,
):
    """Stacked channel panels composite ionic `charge` clouds, velocity streaks from `speeds`, and `potentials` contours between charged walls."""
    points, opened = opened_strip(topology, vertices, n_across, period)
    camera = strip_camera(vertices, period, n_pixels)
    layers_per_strip = liquid_layers(points, opened, n_across, charge, speeds, potentials, falls, camera)
    return strip_stack([
        white_composite(drawing.composited(layers + [wall_strokes(points, n_across, charges, camera)]))
        for layers, charges in zip(layers_per_strip, wall_charges)
    ])


def pumping_animation(
    topology: Topology,
    vertices: Extensor,
    n_across: int,
    period: float,
    charge: Extensor,
    speeds: Extensor,
    potentials: Extensor,
    falls: Extensor,
    wall_charges: np.ndarray,
    frames: Iterator[Extensor],
) -> Iterator[np.ndarray]:
    """Yields stacked RGB frames of neutral dye stripes filtering light above the steady double-layer and electro-osmotic flow background."""
    points, opened = opened_strip(topology, vertices, n_across, period)
    V_opened, E_opened, F_opened = opened.n_elements
    camera = strip_camera(vertices, period, n_pixels)
    backgrounds = [
        white_composite(drawing.composited(layers)) / 255
        for layers in liquid_layers(points, opened, n_across, charge, speeds, potentials, falls, camera)
    ]
    walls = [wall_strokes(points, n_across, charges, camera) for charges in wall_charges]
    gap = np.full((gap_pixels, camera.shape[1], 3), 255, dtype=np.uint8)
    for dye in frames:
        shares = np.clip(seam_values(dye.to_array(), n_across), 0, 1)         # [..., V_opened]
        panels = []
        for share, background, wall in zip(shares, backgrounds, walls):
            filtering = drawing.composited([
                field_layer(points, opened, np.zeros(V_opened), 1 - share[:, None] * (1 - pumped_colour), camera)])
            transmitted = background * (filtering[..., :3] + (1 - filtering[..., 3:]))
            opaque = np.concatenate([transmitted, np.ones(transmitted.shape[:-1] + (1,))], axis=-1)
            panels += [white_composite(drawing.composited([opaque, wall])), gap]
        yield np.concatenate(panels[:-1])


def heat_figure(
    topology: Topology,
    vertices: Extensor,
    n_across: int,
    period: float,
    heat: Extensor,
    potentials: Extensor,
    falls: Extensor,
    wall_charges: np.ndarray,
):
    """Stacked channel panels shade local entropy-production `heat` beneath `potentials` contours and charged wall strokes."""
    points, opened = opened_strip(topology, vertices, n_across, period)
    camera = strip_camera(vertices, period, n_pixels)
    values = seam_values(heat.batch().to_array(), n_across)
    hottest = values.max()
    contour_layers = equipotential_lines(points, opened, potentials, falls, n_across, camera)
    return strip_stack([
        white_composite(drawing.composited([
            field_layer(points, opened, value, heat_map(value / hottest)[:, :3], camera),
            lines,
            wall_strokes(points, n_across, charges, camera),
        ]))
        for value, lines, charges in zip(values, contour_layers, wall_charges)
    ])


@style()
def efficiency_figure(half_widths: np.ndarray, surface_charges: np.ndarray, efficiency: Extensor):
    """Logarithmic filled contours plot peak electrokinetic generator `efficiency` over slit `half_widths` and wall `surface_charges`."""
    values = efficiency.to_array().reshape(len(half_widths), len(surface_charges)).T
    magnitudes = np.abs(surface_charges)
    figure, ax = subplots(efficiency_width, efficiency_height)
    shades = efficiency_map(np.linspace(*efficiency_shades, 14))
    ax.contourf(half_widths, magnitudes, values, levels=np.linspace(0, values.max(), 15), colors=shades)
    lines = ax.contour(half_widths, magnitudes, values, levels=efficiency_levels, colors=[wall_colour], linewidths=1.0)
    ax.clabel(lines, fmt=lambda level: f"{100 * level:.0f}%")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("half width / Debye length")
    ax.set_ylabel("wall charge")
    figure.tight_layout(pad=0.4)
    return figure


@style()
def profiles_figure(topology: Topology, vertices: Extensor, n_across: int, velocities: Extensor):
    """Cross-channel curves compare pressure-driven and field-driven axial `velocities` (`[2] Point-[F]`) normalized by their station mean."""
    columns = topology.elements[2] // n_across
    station = (columns.min(axis=-1) == 0) & (columns.max(axis=-1) == 1)
    centres = topology.incidence[2, 0].mean(axis=0) * vertices                # Point-[F]
    places = plane_readings(mv.y, centres)[station]
    order = np.argsort(places)
    half_width = np.abs(plane_readings(mv.y, vertices)).max()
    # Zero-slip wall endpoints bracket the sorted triangle-centroid speeds across the widest station.
    places = np.concatenate([[-half_width], places[order], [half_width]])
    speeds = np.pad(plane_readings(mv.x, velocities)[:, station][:, order], ((0, 0), (1, 1)))
    speeds = speeds / (np.trapezoid(speeds, places, axis=-1) / (2 * half_width))[:, None]
    figure, ax = subplots(profile_width, profile_height)
    for speed, other, colour, label in zip(speeds, speeds[::-1], profile_colours, profile_labels):
        ax.plot(places, speed, color=colour)
        ahead = np.argmax(speed - other)
        alignment = ("left", "center", "right")[np.digitize(places[ahead], [-half_width / 2, half_width / 2])]
        ax.annotate(
            label, (places[ahead], speed.max()), xytext=(0, 4), textcoords="offset points",
            ha=alignment, va="bottom", color=colour,
        )
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["bottom"].set_position(("data", 0))
    ax.set(
        xlim=(-half_width, half_width),
        ylim=(0, 1.15 * speeds.max()),
        xticks=[-half_width, 0, half_width],
        yticks=profile_speeds,
        xlabel="across / Debye length",
        ylabel="velocity / mean",
    )
    figure.tight_layout(pad=0.4)
    return figure


def dyes_animation(
    topology: Topology,
    vertices: Extensor,
    n_across: int,
    period: float,
    wall_charges: np.ndarray,
    charge: Extensor,
    flows: Extensor,
    potentials: Extensor,
    falls: Extensor,
    frames: Iterator[Extensor],
) -> Iterator[np.ndarray]:
    """Yields stacked two-strip RGB frames comparing neutral dye advection under pressure drop against neutral and oppositely charged dyes under an electric field."""
    points, opened = opened_strip(topology, vertices, n_across, period)
    V_opened, E_opened, F_opened = opened.n_elements
    widest = np.abs(plane_readings(mv.y, vertices)).max()
    camera = Camera(
        mv.x, mv.y, 0 * mv.x, mv.w, 0.0,
        np.array([period / 2, 0.0]), np.array([period / 2, widest * (1 + dye_margin)]), n_pixels,
    )
    walls = wall_strokes(points, n_across, wall_charges, camera)
    colours = np.array([neutral_colour, neutral_colour, positive_colour, negative_colour])  # [4, 3]
    panel_runs = [[0], [1, 2, 3]]
    density = seam_values(charge.to_array(), n_across)                        # [V_opened]
    glow = 1 - glow_strength * (1 - charge_map(0.5 + density / (2 * np.abs(density).max()))[:, :3])
    fastest = np.abs(plane_readings(mv.x, flows)).max()
    velocities = flows.batch()                                                # [2, V] Point-
    closed_velocities = concatenate([velocities, velocities[:, :n_across]], axis=-1)  # [2, V_opened] Point-
    backgrounds = [
        white_composite(drawing.composited([
            field_layer(points, opened, density, glow, camera),
            flow.layer(points, opened, velocity, camera, fastest, streak_colour) * np.array([1, 1, 1, streak_ink]),
            lines,
        ])) / 255
        for velocity, lines in zip(
            closed_velocities, equipotential_lines(points, opened, potentials, falls, n_across, camera))
    ]
    gap = np.full((gap_pixels, camera.shape[1], 3), 255, dtype=np.uint8)
    for dye in frames:
        shares = np.clip(seam_values(dye.to_array(), n_across) / dye_full, 0, 1)  # [4, V_opened]
        panels = []
        for run_indices, background in zip(panel_runs, backgrounds):
            transmittance = np.clip(
                1 - (shares[run_indices, :, None] * (1 - colours[run_indices, None, :])).sum(axis=0), 0, 1)
            filtering = drawing.composited([field_layer(points, opened, np.zeros(V_opened), transmittance, camera)])
            transmitted = background * (filtering[..., :3] + (1 - filtering[..., 3:]))
            opaque = np.concatenate([transmitted, np.ones(transmitted.shape[:-1] + (1,))], axis=-1)
            panels += [white_composite(drawing.composited([opaque, walls])), gap]
        yield np.concatenate(panels[:-1])
