"""Renders triangulated Hall bars, Corbino rings, carrier flux turns, and transport-coefficient curves for two-carrier magnetotransport.

Cameras read conductor vertices through the spatial planes `x` and `y` (rotated by a quarter turn so the Hall bar stands upright) and weigh each point by `w & point`. Temperature tints each triangle linearly between cool blue, equilibrium grey, and warm red, overlaid with electrochemical equipotential contours, feathered white and black drift streaks for the two carrier species, and gold contacts. Between the two conductors, a disc in the spatial plane scales its area with the magnetic field strength and carries a rim chevron showing the turning sense.

Flux panels pair charge fluxes against `xt` and `yt`, entropy fluxes against `xs` and `ys`, and electric drives against `axis & time`.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator

from numga import concatenate, stack
from numga.extensor import Extensor
from numga.render import Camera, contours, drawing, flow, simplices
from numga.topology import Topology
from examples.thermodynamics.magnetotransport import core
from examples.util import charts
from examples.util.panels import image_figure, offset_panel, overlay

mv = core.mv
State = tuple[np.ndarray, core.Potential, core.Spatial, core.Potential, core.Spatial]
Frame = tuple[core.Magnetic, State, State]
DrivenState = tuple[np.ndarray, core.Potential, core.Spatial]

# Temperature palette, reach under electric current drive, rest fill, border strokes, and gold contact strokes.
temperature_colours = LinearSegmentedColormap.from_list(
    "temperature", ((0.0, "#2f6eb5"), (0.5, "#7d8794"), (0.75, "#e2913f"), (1.0, "#c0391d")))
electric_reach = 0.5
rest_colour = np.array(to_rgb("#dde2e8"))
outline_colour, outline_width = np.array(to_rgb("#4b4f55")), 1.5
contact_colour, contact_width = np.array(to_rgb("#b8860b")), 3.0

# Equipotential contour styling and reference step count.
level_colour, level_ink, level_width = np.array(to_rgb("#1d2228")), 0.55, 1.0
n_levels = 6

# Drift streak colours, seed spacing, speck radius, and reference pixel lengths.
streak_colours = np.array([to_rgb("#f8f8f8"), to_rgb("#15181c")])
streak_grains, streak_speck, streak_ink = np.array([20.0, 23.0]), 1.4, 1.0
bar_streak_length, ring_streak_length = 45.0, 40.0

# Magnetic field disc and sense chevron geometry.
magnetic_colour, patch_opacity = np.array(to_rgb("dodgerblue")), 0.8
patch_reach = 0.22
arc_reach = 1.25
arc_start, arc_span = np.radians(-60.0), np.radians(240.0)
n_arc = 32
arc_width, arc_chevron = 1.8, 8.0

# Framing margin, inter-panel gap, and quarter-turn rotor orienting the Hall bar upright.
margin, gap = 0.08, 8
upright = (mv.xy * (np.pi / 4)).exp()                                         # [] Rotor+

# Flux turn diagram dimensions and stroke styling.
turn_size, turn_across, turn_up = (220, 150), (-0.35, 1.1), (-0.5, 0.5)
electric_colour = np.array(to_rgb("crimson"))
arrow_width, total_width, arrow_chevron = 2.0, 3.0, 9.0
electron_arrow_colour, total_colour = np.array(to_rgb("#9aa3ad")), np.array(to_rgb("#6d28d9"))
sweep_dots = (1.5, 3.0)

# Transport-coefficient chart dimensions, colours, and balance axis labels.
chart_size = (430, 460)
spin_chart_size = (430, 170)
along_colour, across_colour, heat_colour = "#4f5966", "#c2410c", "#1d4ed8"
balance_ticks = [-1, 0, 1]
balance_names = ["electrons", "compensated", "holes"]
magnetization_names = ["spin down", "unmagnetized", "spin up"]


# --- plumbing -------------------------------------------------------------------------
def white_composite(picture: np.ndarray) -> np.ndarray:
    """Alpha-compositing `picture` (`[..., 4]`) over white yields RGB pixel bytes (`[..., 3]`)."""
    composited_rgb = picture[..., :3] + (1 - picture[..., 3:])
    return np.round(np.clip(composited_rgb, 0, 1) * 255).astype(np.uint8)


def plane_readings(plane: Extensor, values: Extensor) -> np.ndarray:
    """Pairing `plane` against `values` extracts scalar coordinate readings in the batch shape of `values`."""
    return (plane & values.batch()).to_array().reshape(values.batch().shape)


def potential_readings(potentials: Extensor, quantity: core.Parcel) -> np.ndarray:
    """Pairing `potentials` (`[...] Potential+[V]`) with a unit `quantity` (`Parcel-`) extracts vertex electrochemical potentials or temperatures."""
    return (potentials.batch() & quantity).to_array().reshape(potentials.batch().shape)


def temperature_tints(temperatures: np.ndarray, reach: float) -> np.ndarray:
    """Mapping `temperatures` (`[V]`) onto `temperature_colours` within `+-reach` around equilibrium grey produces vertex RGB values (`[V, 3]`)."""
    return temperature_colours(np.clip(0.5 + temperatures / (2 * reach), 0, 1))[:, :3]


def conductor_camera(vertices: core.Point, across_plane: core.Plane, up_plane: core.Plane, resolution: float) -> Camera:
    """Frames `vertices` (`Point-[V]`) with `margin` padding in a `Camera` paired along `across_plane` and `up_plane` at `resolution` pixels per unit."""
    coordinates = np.stack([plane_readings(across_plane, vertices), plane_readings(up_plane, vertices)], axis=-1)   # [V, 2]
    lower_bound, upper_bound = coordinates.min(axis=0) - margin, coordinates.max(axis=0) + margin
    return Camera(across_plane, up_plane, 0 * mv.x, mv.w, 0.0, (lower_bound + upper_bound) / 2,
                  (upper_bound - lower_bound) / 2, round((upper_bound - lower_bound)[0] * resolution))


def patch_camera(height: int, resolution: float) -> Camera:
    """Frames the magnetic-field indicator strip around the origin at `height` pixels tall and `resolution` pixels per bar width."""
    pixels = round(2 * (arc_reach * patch_reach + margin) * resolution)
    return Camera(mv.x, mv.y, 0 * mv.x, mv.w, 0.0, np.zeros(2), np.array([pixels, height]) / (2 * resolution), pixels)


def conductor_outline(topology: Topology, vertices: core.Point, contacts: tuple[np.ndarray, np.ndarray],
                      camera: Camera) -> list:
    """Builds border strokes and gold contact strokes for a triangulated conductor."""
    points = vertices.batch()
    contact_edges = np.isin(topology.elements[1], np.concatenate(contacts)).all(axis=-1)   # [E]
    return [simplices.strokes(points, topology.border, outline_colour, outline_width, camera),
            simplices.strokes(points, topology.subset(contact_edges, 1), contact_colour, contact_width, camera)]


def field_patch(centre: core.Point, magnetic: core.Magnetic, reference: core.Magnetic, radius: float,
                camera: Camera) -> list:
    """Draws `magnetic` (`[] Magnetic+`) relative to `reference` as an area-scaled disc with a sense-indicating rim chevron."""
    share = (magnetic.scalar_product(~reference) / reference.scalar_norm_squared()).to_array()
    size = radius * np.sqrt(np.abs(share))
    angles = np.sign(share) * (arc_start + np.linspace(0, arc_span, n_arc))   # [n_arc]
    turns = (reference.normalized() * (-angles / 2)).exp()                    # [n_arc] Rotor+
    arc_points = centre + arc_reach * size * (turns >> core.x_direction)      # [n_arc] Point-
    tip_mask = np.arange(n_arc - 1) == n_arc - 2                              # [n_arc - 1]
    return [simplices.markers(stack([centre]), magnetic_colour, size / camera.pixel, camera, opacity=patch_opacity),
            simplices.strokes(arc_points, Topology.chains([n_arc]), magnetic_colour, arc_width, camera,
                              chevrons=tip_mask, chevron=arc_chevron)]


def face_corners(topology: Topology, vertices: core.Point) -> tuple[core.Point, Topology]:
    """Splits `topology` into disconnected triangles (`[3 * F] Point-`) so piecewise-constant triangle velocities render cleanly."""
    V, E, F = topology.n_elements
    corners = (topology.incidence[2, 0] * vertices).batch().reshape(-1)       # [3 * F] Point-
    return corners, Topology(3 * F, np.arange(3 * F).reshape(3, F).T)


def carrier_speeds(velocities: core.Spatial) -> np.ndarray:
    """Antinorm of spatial drift `velocities` (`[2] Spatial-[F]`) gives scalar carrier speeds (`[2, F]`)."""
    sizes = velocities.batch().antinorm()                                     # [2, F] Scalar+
    return sizes.to_array()


def contact_voltage(potentials: core.Potential, contacts: tuple[np.ndarray, np.ndarray]) -> float:
    """Difference in electrochemical potential between the first vertices of the two contact sets gives the terminal voltage."""
    values = potential_readings(potentials, core.time)
    return values[contacts[0][0]] - values[contacts[1][0]]


def picture_row(pictures: list[np.ndarray]) -> np.ndarray:
    """Vertically centres RGB `pictures` on white and concatenates them left to right."""
    height = max(picture.shape[0] for picture in pictures)
    levelled_panels = []
    for picture in pictures:
        canvas = np.full((height, picture.shape[1], 3), 255, dtype=np.uint8)
        top_offset = (height - picture.shape[0]) // 2
        canvas[top_offset:top_offset + picture.shape[0]] = picture
        levelled_panels.append(canvas)
    return np.concatenate(levelled_panels, axis=1)


def conductor_layers(topology: Topology, vertices: core.Point, contacts: tuple[np.ndarray, np.ndarray],
                     potentials: core.Potential, velocities: core.Spatial, shares: np.ndarray, reach: float,
                     spacing: float, reference_speed: float, streak_length: float, camera: Camera) -> list:
    """Assembles temperature tint, electrochemical equipotential contours, two-species velocity streaks, and gold contacts for one conductor."""
    points = vertices.batch()
    face_places, split_topology = face_corners(topology, vertices)
    temperatures = potential_readings(potentials, core.entropy)               # [V]
    lines = contours.Lines(spacing, spacing / 2, np.append(level_colour, level_ink), level_width)
    corner_velocities = [concatenate([velocities[kind].batch()] * 3) for kind in range(len(streak_colours))]
    return [contours.picture(points, topology, temperatures, temperature_tints(temperatures, reach), camera),
            contours.picture(points, topology, potential_readings(potentials, core.time), np.zeros(3), camera,
                             lines=(lines,), opacity=0.0),
            *[flow.layer(face_places, split_topology, velocity, camera, reference_speed, colour, length=streak_length,
                         grain_pixels=grain, speck_pixels=streak_speck, ink_opacity=streak_ink * share)
              for velocity, colour, grain, share in zip(corner_velocities, streak_colours, streak_grains, shares)],
            *conductor_outline(topology, vertices, contacts, camera)]


def side_by_side_frames(bar_topology: Topology, bar_vertices: core.Point, bar_contacts: tuple[np.ndarray, np.ndarray],
                        ring_topology: Topology, ring_vertices: core.Point, ring_contacts: tuple[np.ndarray, np.ndarray],
                        bar_resolution: float, ring_resolution: float, reference: core.Magnetic,
                        scales: list[tuple[float, float, float]],
                        frames: Iterable[tuple[core.Magnetic, DrivenState, DrivenState]]) -> Iterator[np.ndarray]:
    """Yields side-by-side RGB frames of the upright Hall bar, magnetic-field indicator, and Corbino ring."""
    bar_cam = conductor_camera(bar_vertices, upright >> mv.x, upright >> mv.y, bar_resolution)
    ring_cam = conductor_camera(ring_vertices, mv.x, mv.y, ring_resolution)
    field_cam = patch_camera(max(bar_cam.shape[0], ring_cam.shape[0]), bar_resolution)
    # Pair each conductor's topology, contacts, camera, and reference streak length.
    views = ((bar_topology, bar_vertices, bar_contacts, bar_cam, bar_streak_length),
             (ring_topology, ring_vertices, ring_contacts, ring_cam, ring_streak_length))
    for magnetic, *driven_states in frames:
        pictures = []
        for view, (reach, spacing, reference_speed), (densities, potentials, velocities) in zip(
                views, scales, driven_states):
            topology, vertices, contacts, camera, streak_length = view
            shares = densities / densities.max()
            pictures.append(white_composite(drawing.composited(
                conductor_layers(topology, vertices, contacts, potentials, velocities, shares, reach, spacing,
                                 reference_speed, streak_length, camera))))
        bar_picture, ring_picture = pictures
        patch_picture = white_composite(drawing.composited(
            field_patch(core.origin, magnetic, reference, patch_reach, field_cam)))
        yield picture_row([bar_picture, patch_picture, ring_picture])


# --- drawing --------------------------------------------------------------------------
def conductors_figure(bar_topology: Topology, bar_vertices: core.Point, bar_contacts: tuple[np.ndarray, np.ndarray],
                      ring_topology: Topology, ring_vertices: core.Point, ring_contacts: tuple[np.ndarray, np.ndarray],
                      bar_resolution: float, ring_resolution: float) -> Figure:
    """Renders the Hall bar and Corbino ring side by side with pale interiors and gold contacts."""
    pictures = []
    for topology, vertices, contacts, resolution in ((bar_topology, bar_vertices, bar_contacts, bar_resolution),
                                                     (ring_topology, ring_vertices, ring_contacts, ring_resolution)):
        V, E, F = topology.n_elements
        camera = conductor_camera(vertices, mv.x, mv.y, resolution)
        fill = contours.picture(vertices.batch(), topology, np.zeros(V), rest_colour, camera)
        pictures.append(white_composite(drawing.composited([fill, *conductor_outline(topology, vertices, contacts, camera)])))
    bar_picture, ring_picture = pictures
    return image_figure(picture_row([bar_picture, np.full((1, gap, 3), 255, dtype=np.uint8), ring_picture]))


def current_driven_animation(bar_topology: Topology, bar_vertices: core.Point,
                             bar_contacts: tuple[np.ndarray, np.ndarray], ring_topology: Topology,
                             ring_vertices: core.Point, ring_contacts: tuple[np.ndarray, np.ndarray],
                             bar_resolution: float, ring_resolution: float, reference: Frame, frames: Iterable[Frame]
                             ) -> Iterator[np.ndarray]:
    """Yields side-by-side RGB frames of the isothermal current-driven Hall bar and Corbino ring scaled to `reference`."""
    field, *conductors = reference
    scales = [(electric_reach, abs(contact_voltage(electric, contacts)) / n_levels, np.median(carrier_speeds(drifts)))
              for (_, electric, drifts, _, _), contacts in zip(conductors, (bar_contacts, ring_contacts))]
    driven_states = ((magnetic, *[(densities, electric, drifts) for densities, electric, drifts, _, _ in states])
                     for magnetic, *states in frames)
    return side_by_side_frames(bar_topology, bar_vertices, bar_contacts, ring_topology, ring_vertices, ring_contacts,
                               bar_resolution, ring_resolution, field, scales, driven_states)


def heat_driven_animation(bar_topology: Topology, bar_vertices: core.Point,
                          bar_contacts: tuple[np.ndarray, np.ndarray], ring_topology: Topology,
                          ring_vertices: core.Point, ring_contacts: tuple[np.ndarray, np.ndarray],
                          bar_resolution: float, ring_resolution: float, reference: Frame, frames: Iterable[Frame]
                          ) -> Iterator[np.ndarray]:
    """Yields side-by-side RGB frames of the open-circuit heat-driven Hall bar and Corbino ring scaled to `reference`."""
    field, *conductors = reference
    scales = [(abs(potential_readings(thermal, core.entropy)[contacts[0][0]]),
               abs(contact_voltage(thermal, contacts)) / n_levels, np.median(carrier_speeds(drifts)))
              for (_, _, drifts, thermal, _), contacts in zip(conductors, (bar_contacts, ring_contacts))]
    driven_states = ((magnetic, *[(densities, thermal, drifts) for densities, _, _, thermal, drifts in states])
                     for magnetic, *states in frames)
    return side_by_side_frames(bar_topology, bar_vertices, bar_contacts, ring_topology, ring_vertices, ring_contacts,
                               bar_resolution, ring_resolution, field, scales, driven_states)


@charts.style()
def turns_figure(drive: core.Drive, fluxes: core.Flux) -> Figure:
    """Pure-electron, pure-hole, and compensated fluxes (`[2] Flux-`) driven by `drive` (`[] Drive+`) appear in charge and entropy readout planes."""
    local_mv = drive.context.multivector
    x_dir, y_dir, time_dir = local_mv.x.dual(), local_mv.y.dual(), -local_mv.t.dual()
    figure, axes = charts.subplots(2 * turn_size[0], turn_size[1], ncols=2)
    for ax in axes:
        ax.set_xlim(*turn_across)
        ax.set_ylim(*turn_up)
        ax.set_aspect("equal")
        ax.set_axis_off()
    figure.tight_layout(pad=0.2)
    electrons, holes = fluxes
    compensated_flux = (electrons + holes) / 2                                # [] Flux-
    arrows = stack((0 * fluxes, fluxes), axis=-1).reshape(-1)                 # [4] Flux-
    for ax, planes in zip(axes, ((local_mv.xt, local_mv.yt), (local_mv.xs, local_mv.ys))):
        field_view = offset_panel(ax, x_dir & time_dir, y_dir & time_dir)
        flux_view = offset_panel(ax, *planes)
        overlay(ax, [
            simplices.strokes(stack((0 * drive, drive)), Topology.chains([2]), electric_colour, arrow_width,
                              field_view, chevrons=np.ones(1), chevron=arrow_chevron),
            simplices.strokes(fluxes, Topology.chains([2]), np.zeros(3), charts.LINE_PX, flux_view, dashes=sweep_dots,
                              cap=simplices.Cap.ROUND, opacity=0.5),
            simplices.strokes(arrows, Topology.chains([2, 2]), np.stack([electron_arrow_colour, streak_colours[1]]),
                              arrow_width, flux_view, chevrons=np.ones(2), chevron=arrow_chevron),
            simplices.strokes(stack((0 * compensated_flux, compensated_flux)), Topology.chains([2]), total_colour,
                              total_width, flux_view, chevrons=np.ones(1), chevron=arrow_chevron)])
    return figure


@charts.style()
def readings_figure(balances: np.ndarray, resistance: core.Scalar, hall: core.Scalar, seebeck: core.Scalar,
                    nernst: core.Scalar, righi_leduc: core.Scalar, names: list[str]) -> Figure:
    """Plots longitudinal and Hall resistances, Seebeck and Nernst signals, and the Righi-Leduc temperature ratio (`[n_frames] Scalar+`) against carrier `balances`."""
    figure, axes = charts.subplots(*chart_size, nrows=3, sharex=True)
    for ax, curves, label, spacing in zip(axes, (((resistance, along_colour, "along"), (hall, across_colour, "Hall")),
                                                 ((seebeck, along_colour, "Seebeck"), (nernst, across_colour, "Nernst")),
                                                 ((righi_leduc, heat_colour, "Righi–Leduc"),)),
                                          ("resistance", "thermopower", "temperature\nacross"), (1, 1, 0.1)):
        for values, colour, name in curves:
            curve_values = values.to_array().reshape(-1)
            ax.plot(balances, curve_values, color=colour)
            ax.annotate(name, (balances[-1], curve_values[-1]), xytext=(4, 0), textcoords="offset points", va="center",
                        color=colour)
        ax.axhline(0.0, color="#555555", linewidth=charts.FRAME_WIDTH)
        ax.spines[["top", "right", "bottom"]].set_visible(False)
        ax.set(xlim=(balances[0], balances[-1]), ylabel=label)
        ax.yaxis.set_major_locator(MultipleLocator(spacing))
    axes[-1].set(xticks=balance_ticks, xticklabels=names)
    figure.tight_layout(pad=0.4)
    return figure


@charts.style()
def spin_readings_figure(magnetizations: np.ndarray, resistance: core.Scalar, hall: core.Scalar) -> Figure:
    """Plots longitudinal magnetoresistance and anomalous Hall resistance (`[n_frames] Scalar+`) against `magnetizations` at zero external magnetic field."""
    figure, ax = charts.subplots(*spin_chart_size)
    for values, colour, name in ((resistance, along_colour, "along"), (hall, across_colour, "Hall")):
        curve_values = values.to_array().reshape(-1)
        ax.plot(magnetizations, curve_values, color=colour)
        ax.annotate(name, (magnetizations[-1], curve_values[-1]), xytext=(4, 0), textcoords="offset points", va="center",
                    color=colour)
    ax.axhline(0.0, color="#555555", linewidth=charts.FRAME_WIDTH)
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.set(xlim=(magnetizations[0], magnetizations[-1]), ylabel="resistance", xticks=balance_ticks,
           xticklabels=magnetization_names)
    ax.yaxis.set_major_locator(MultipleLocator(1))
    figure.tight_layout(pad=0.4)
    return figure
