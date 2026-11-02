"""Drawing for electrokinetic transport on strips of triangles.

A strip closed on itself along its length is opened at its seam to be drawn: its first column again a period further
along, and the triangles across the seam joined to that copy. Its points are read by the planes x and y and weighed by
their volume, `mv.v & point`, the algebra's own projective weight. The mesh is drawn as strokes of its edges; a field
over it as colours on its triangles, a charge density or a heat from white; the flow as streaks of ink along the velocity, read by the same planes, each as
long as the liquid there is fast and fading downstream; and the walls as strokes coloured by their charge, magenta where
positive and green where negative. The best efficiency of slits as generators is a chart.
"""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import to_rgb

from numga import concatenate
from numga.extensor import Extensor
from examples.util.charts import style, subplots
from examples.util.wgsl import contours, flow, image_figure, simplices
from examples.util.wgsl.camera import Camera

Topology = simplices.Topology
wall_colour = np.array(to_rgb("#3b4a57"))
mesh_colour = np.array(to_rgb("#9aa5ae"))
positive_colour, negative_colour = np.array(to_rgb("#c51b7d")), np.array(to_rgb("#4d9221"))
streak_colour = np.array(to_rgb("#2b2f33"))
# How much of the streaks' ink shows, so that the colours beneath read through it.
streak_ink = 0.6
# A neutral dye's colour; a charged dye takes its charge's colour, as the walls do.
neutral_colour = np.array(to_rgb("#1f4e79"))
# The room above and below a dyed channel's walls, as a share of its half width; the share of a band's starting
# strength at which a dye shows its full colour, about what a thin band keeps once it has spread.
dye_margin, dye_full = 0.06, 0.4
# How much of the charge's colour shows in a dyed channel, a faint glow of its double layers.
glow_strength = 0.5
efficiency_map, heat_map, charge_map = (colormaps[name] for name in ("Blues", "Oranges", "PiYG_r"))
# The share of the efficiency's colour map used, light to mid blue, so that its labels read on it; its labelled levels.
efficiency_shades = (0.03, 0.6)
efficiency_levels = np.array([0.01, 0.02, 0.04, 0.06])
# The chart's pixels.
efficiency_width, efficiency_height = 320, 260
# The velocity profiles' chart: its pixels, its curves' colours and labels, the pressure drop's flow and the field's, and
# the speeds marked along it, nothing, the mean and Poiseuille's peak at half again the mean.
profile_width, profile_height = 300, 220
profile_colours = (np.array(to_rgb("#b35806")), neutral_colour)
profile_labels = ("pressure drop", "field")
profile_speeds = (0, 1, 1.5)
# A strip's picture: pixels across, the room above and below its widest place, and the pixels between strips; the widths
# of the walls and the mesh's edges; how much of a field's colour shows at its largest.
n_pixels, margin, gap_pixels = 300, 0.4, 8
wall_width, mesh_width = 3.0, 1.5
# A mesh's picture gets the pixels its rows of vertices need to show apart, within these bounds.
row_pixels, mesh_pixels = 4.0, (300, 700)
field_strength = 0.85


# --- plumbing -------------------------------------------------------------------------
def on_white(picture: np.ndarray) -> np.ndarray:
    """Colour and coverage `[rows, columns, 4]` laid over white, as bytes `[rows, columns, 3]`."""
    shown = picture[..., :3] + (1 - picture[..., 3:])
    return np.round(np.clip(shown, 0, 1) * 255).astype(np.uint8)


def readings(plane: Extensor, values: Extensor) -> np.ndarray:
    """Values read by a plane, as numbers in their batch and field's shape."""
    return (plane & values.batch()).to_array().reshape(values.batch().shape)


def unrolled(topology: Topology, vertices: Extensor, n_across: int, period: float) -> tuple[Extensor, Topology]:
    """A strip closed on itself, opened at its seam for drawing: its vertices `Point-[vertices]` with its first column
    again a period further along, and its triangles, those across the seam joined to that copy."""
    faces = topology.elements[2]
    n_vertices = topology.n_vertices
    columns = faces // n_across
    across_seam = (columns.max(axis=-1) - columns.min(axis=-1) > 1)[:, None] & (columns == 0)
    faces = np.where(across_seam, n_vertices + faces % n_across, faces)
    points = vertices.batch()                                                  # [vertices] Point-
    first = points[:n_across] + vertices.context.multivector.x.dual() * period  # [across] Point-
    return concatenate([points, first]), Topology(n_vertices + n_across, faces)


def strip_camera(vertices: Extensor, period: float, pixels: int) -> Camera:
    """The camera looking down on a strip, `pixels` across: points read by the planes x and y, weighed by their
    volume."""
    mv = vertices.context.multivector
    widest = np.abs(readings(mv.y, vertices)).max()
    return Camera(mv.x, mv.y, 0 * mv.x, mv.v, 0.0, np.array([period / 2, 0.0]),
                  np.array([period / 2, widest + margin]), pixels)


def wall_strokes(points: Extensor, n_across: int, wall_charges: np.ndarray, camera: Camera):
    """A strip's walls, opened at the seam `[columns + 1, across] Point-`, their first and last vertices, each edge
    coloured by the charge at its middle `[columns]`."""
    rows = points.reshape(-1, n_across)
    closed = np.append(wall_charges, wall_charges[:1])
    edge_colours = np.where(((closed[:-1] + closed[1:]) / 2)[:, None] > 0, positive_colour, negative_colour)
    n_points = rows.shape[0]
    return simplices.strokes(concatenate([rows[:, 0], rows[:, -1]]), Topology.chains([n_points, n_points]),
                             np.concatenate([edge_colours, edge_colours]), wall_width, camera)


def shaded(points: Extensor, faces: Topology, values: np.ndarray, colours: np.ndarray, camera: Camera):
    """A field over the opened strip `[vertices + across]`, in its colours."""
    return contours.picture(points, faces, values, colours, camera)


def stacked(images: list[np.ndarray]):
    """Pictures one above the other, a gap between them, as a figure."""
    gap = np.full((gap_pixels, images[0].shape[1], 3), 255, dtype=np.uint8)
    return image_figure(np.concatenate([part for image in images for part in (image, gap)][:-1]))


def closing(values: np.ndarray, n_across: int) -> np.ndarray:
    """Values at a strip's vertices `[..., vertices]`, with its first column's again for the copy at the seam."""
    return np.concatenate([values, values[..., :n_across]], axis=-1)


def draw_mesh(topology: Topology, vertices: Extensor, n_across: int, period: float, wall_charges: np.ndarray):
    """A strip's triangles as strokes of their edges, between its walls coloured by their charge `[columns]`."""
    points, faces = unrolled(topology, vertices, n_across, period)
    # The narrowest place's rows of vertices, their spacing across.
    spacing = 2 * np.abs(readings(vertices.context.multivector.y, vertices)).reshape(-1, n_across)[:, 0].min() / (n_across - 1)
    camera = strip_camera(vertices, period, int(np.clip(row_pixels * period / spacing, *mesh_pixels)))
    layers = [simplices.strokes(points, faces, mesh_colour, mesh_width, camera),
              wall_strokes(points, n_across, wall_charges, camera)]
    return image_figure(on_white(simplices.composited(layers)))


def draw_flow(topology: Topology, vertices: Extensor, n_across: int, period: float, charge: Extensor,
              speeds: Extensor, wall_charges: np.ndarray):
    """Strips one above the other, each coloured by its liquid's charge density `[strips] Scalar+[vertices]`, magenta
    where positive and green where negative on one scale for all, and streaked along its velocity `[strips]
    Point-[vertices]`, a streak as long as the liquid there is fast and fading the way it goes, on one scale for all,
    between walls coloured by their charge `[strips, columns]`."""
    mv = vertices.context.multivector
    points, faces = unrolled(topology, vertices, n_across, period)
    camera = strip_camera(vertices, period, n_pixels)
    densities = closing(charge.batch().to_array(), n_across)
    largest = np.abs(densities).max()
    fastest = np.abs(readings(mv.x, speeds)).max()
    # The velocities as they are, with the first column's again at its copy across the seam, read by the camera's
    # planes in the streaks' shader; a streak is as long as the liquid there is fast, the fastest's a set length.
    velocities = speeds.batch()                                                # [strips, vertices] Point-
    closed = concatenate([velocities, velocities[:, :n_across]], axis=-1)     # [strips, vertices + across] Point-
    return stacked([on_white(simplices.composited([
        shaded(points, faces, density, 1 - field_strength * (1 - charge_map(0.5 + density / (2 * largest))[:, :3]),
               camera),
        flow.layer(points, faces, velocity, camera, fastest, streak_colour) * np.array([1, 1, 1, streak_ink]),
        wall_strokes(points, n_across, wall_charge, camera)]))
        for density, velocity, wall_charge in zip(densities, closed, wall_charges)])


def draw_heat(topology: Topology, vertices: Extensor, n_across: int, period: float, heat: Extensor,
              wall_charges: np.ndarray):
    """Strips one above the other, each coloured by the heat made in it `[strips] Scalar+[vertices]`, from white for
    none to orange for the most in any, between walls coloured by their charge `[strips, columns]`."""
    points, faces = unrolled(topology, vertices, n_across, period)
    camera = strip_camera(vertices, period, n_pixels)
    values = closing(heat.batch().to_array(), n_across)
    hottest = values.max()
    return stacked([on_white(simplices.composited([
        shaded(points, faces, value, heat_map(value / hottest)[:, :3], camera),
        wall_strokes(points, n_across, wall_charge, camera)]))
        for value, wall_charge in zip(values, wall_charges)])


@style()
def draw_efficiency(half_widths: np.ndarray, surface_charges: np.ndarray, efficiency: Extensor):
    """The best efficiency of each slit as a generator `[widths, charges] Scalar+`, over its half width and the size of
    its walls' charge, both on logarithmic axes, with level lines labelled in percent."""
    values = efficiency.to_array().reshape(len(half_widths), len(surface_charges)).T
    sizes = np.abs(surface_charges)
    figure, ax = subplots(efficiency_width, efficiency_height)
    shades = efficiency_map(np.linspace(*efficiency_shades, 14))
    ax.contourf(half_widths, sizes, values, levels=np.linspace(0, values.max(), 15), colors=shades)
    lines = ax.contour(half_widths, sizes, values, levels=efficiency_levels, colors=[wall_colour], linewidths=1.0)
    ax.clabel(lines, fmt=lambda level: f"{100 * level:.0f}%")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("half width / Debye length")
    ax.set_ylabel("wall charge")
    figure.tight_layout(pad=0.4)
    return figure


@style()
def draw_profiles(topology: Topology, vertices: Extensor, n_across: int, velocities: Extensor):
    """The liquid's velocity along a straight strip against the place across it, under the pressure drop and under the
    field `[2] Point-[faces]`, read in the triangles between the strip's first two columns of vertices at their
    centres, and at rest on the walls; each curve labelled at the height of its top, where it runs furthest ahead of
    the other, the axis across ending at the walls and its zero the axis along."""
    mv = vertices.context.multivector
    columns = topology.elements[2] // n_across
    station = (columns.min(axis=-1) == 0) & (columns.max(axis=-1) == 1)
    # Each triangle's centre, the mean of its corners.
    centres = topology.incidence[2, 0].mean(axis=0) * vertices                  # Point-[faces]
    places = readings(mv.y, centres)[station]
    order = np.argsort(places)
    half_width = np.abs(readings(mv.y, vertices)).max()
    # The liquid at rest on the walls, which the triangles' centres stop short of.
    places = np.concatenate([[-half_width], places[order], [half_width]])
    speeds = np.pad(readings(mv.x, velocities)[:, station][:, order], ((0, 0), (1, 1)))   # [2, triangles + 2]
    figure, ax = subplots(profile_width, profile_height)
    for speed, other, colour, label in zip(speeds, speeds[::-1], profile_colours, profile_labels):
        ax.plot(places, speed, color=colour)
        # Aligned away from the nearer wall, so that the label stays within the chart.
        ahead = np.argmax(speed - other)
        alignment = ("left", "center", "right")[np.digitize(places[ahead], [-half_width / 2, half_width / 2])]
        ax.annotate(label, (places[ahead], speed.max()), xytext=(0, 4), textcoords="offset points", ha=alignment,
                    va="bottom", color=colour)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["bottom"].set_position(("data", 0))
    ax.set(xlim=(-half_width, half_width), ylim=(0, 1.15 * speeds.max()), xticks=[-half_width, 0, half_width],
           yticks=profile_speeds, xlabel="across / Debye length", ylabel="velocity / mean")
    figure.tight_layout(pad=0.4)
    return figure


def animate_dyes(topology: Topology, vertices: Extensor, n_across: int, period: float, wall_charges: np.ndarray,
                 charge: Extensor, flows: Extensor, frames) -> Iterator[np.ndarray]:
    """The dye in a channel at every frame as it comes `[runs] Scalar+[vertices]`, two strips one above the other: a
    neutral dye carried by a pressure drop's flow, and below it the same dye carried by a field's electro-osmotic flow
    together with a positive and a negative dye, in their charges' colours, drifting under the field as well. Each
    strip is the liquid: faintly coloured by its charge density `Scalar+[vertices]`, its double layers, and streaked
    along its velocity under the strip's drive `[strips] Point-[vertices]`, on one scale for both; over it each dye
    from nothing where there is none to its colour, and the walls, coloured by their charge `[stations]`."""
    points, faces = unrolled(topology, vertices, n_across, period)
    # The room above and below the walls, a share of the channel's width, so that the walls' strokes show whole.
    mv = vertices.context.multivector
    widest = np.abs(readings(mv.y, vertices)).max()
    camera = Camera(mv.x, mv.y, 0 * mv.x, mv.v, 0.0, np.array([period / 2, 0.0]),
                    np.array([period / 2, widest * (1 + dye_margin)]), n_pixels)
    walls = wall_strokes(points, n_across, wall_charges, camera)
    colours = np.array([neutral_colour, neutral_colour, positive_colour, negative_colour])   # [runs, 3]
    panels = [[0], [1, 2, 3]]
    density = closing(charge.to_array(), n_across)                              # [vertices + across]
    glow = 1 - glow_strength * (1 - charge_map(0.5 + density / (2 * np.abs(density).max()))[:, :3])
    fastest = np.abs(readings(mv.x, flows)).max()
    velocities = flows.batch()                                                  # [strips, vertices] Point-
    closed = concatenate([velocities, velocities[:, :n_across]], axis=-1)      # [strips, vertices + across] Point-
    # Each strip's liquid, its charge's glow under its flow's streaks, laid over white once.
    liquids = [on_white(simplices.composited([
        shaded(points, faces, density, glow, camera),
        flow.layer(points, faces, velocity, camera, fastest, streak_colour) * np.array([1, 1, 1, streak_ink])])) / 255
        for velocity in closed]
    gap = np.full((gap_pixels, camera.shape[1], 3), 255, dtype=np.uint8)
    for dye in frames:
        shares = np.clip(closing(dye.to_array(), n_across) / dye_full, 0, 1)  # [runs, vertices + across]
        pictures = []
        for runs, liquid in zip(panels, liquids):
            # Each dye takes away from the light what its colour lacks, by its share.
            passing = np.clip(1 - (shares[runs, :, None] * (1 - colours[runs, None, :])).sum(axis=0), 0, 1)
            filtering = shaded(points, faces, np.zeros(len(passing)), passing, camera)
            seen = liquid * (filtering[..., :3] + (1 - filtering[..., 3:]))     # [rows, columns, 3]
            opaque = np.concatenate([seen, np.ones(seen.shape[:-1] + (1,))], axis=-1)
            pictures += [on_white(simplices.composited([opaque, walls])), gap]
        yield np.concatenate(pictures[:-1])
