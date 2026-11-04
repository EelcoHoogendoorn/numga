"""Drawing for the Meissner effect: the disk of triangles around the rod, the rod's field lines and supercurrent over
time, as a superconductor and as a perfect conductor side by side, and the field's decay into the cold superconductor
as a chart.

The disk's points are read by the planes x and y; a point of space is the direction from the origin to it, so its
weight is one. The magnetic field's lines are the potential's level lines, and the supercurrent along the rod is red
where it runs out of the picture and blue where into it; the rod's edge is a grey stroke.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure

from numga.extensor import Extensor
from examples.util.charts import style, subplots
from examples.util.wgsl import contours, image_figure, simplices
from examples.util.wgsl.camera import Camera

Topology = simplices.Topology
mesh_colour = np.array(to_rgb("#9aa5ae"))
rod_colour = np.array(to_rgb("#e3e0da"))
edge_colour = np.array(to_rgb("#7d7466"))
line_colour = np.array(to_rgb("#2b2f33"))
current_map, strength_map = colormaps["RdBu_r"], colormaps["Purples"]
measured_colour, london_colour = np.array(to_rgb("#1d4ed8")), np.array(to_rgb("#c2410c"))
# The charts' pixels; a trapped field's panel's pixels across, its current lines' spacing in units of the field's
# strength, and the strength, in flux quanta per squared radius, that saturates its colours.
chart_width, chart_height = 300, 220
ring_pixels, current_spacing, strongest = 130, 0.25, 3.0
# A panel's pixels across; the widths of the mesh's edges, the rod's edge and the field lines; the field lines' spacing
# in the potential, in units of the applied field times a rod radius; the supercurrent that saturates the colours, as
# a share of the applied field over the penetration depth; the pixels between the panels.
n_pixels = 200
mesh_width, edge_width, line_width = 0.75, 1.5, 1.2
line_spacing = 0.2
current_share = 1.5
gap_pixels = 6


# --- plumbing -------------------------------------------------------------------------
def on_white(picture: np.ndarray) -> np.ndarray:
    """Colour and coverage `[rows, columns, 4]` laid over white, as bytes `[rows, columns, 3]`."""
    shown = picture[..., :3] + (1 - picture[..., 3:])
    return np.round(np.clip(shown, 0, 1) * 255).astype(np.uint8)


def disk_camera(vertices: Extensor) -> Camera:
    """The camera looking down on the disk, framing it, `n_pixels` across: points read by the planes x and y."""
    mv = vertices.context.multivector
    across = (mv.x & vertices.batch()).to_array()
    half = np.abs(across).max()
    return Camera(mv.x, mv.y, mv.z, 0 * mv.x, 1.0, np.zeros(2), np.array([half, half]), n_pixels)


def rod_edge(vertices: Extensor, topology: Topology, in_rod: np.ndarray, camera: Camera):
    """The rod's edge as strokes: the border of the triangles whose corners all lie in it `[vertices]`."""
    whole = in_rod[topology.elements[2]].all(axis=-1)
    return simplices.strokes(vertices.batch(), topology.subset(whole, -1).border, edge_colour, edge_width, camera)


# --- drawing --------------------------------------------------------------------------
def draw_rod(topology: Topology, vertices: Extensor, in_rod: np.ndarray) -> Figure:
    """The disk's triangles as strokes of their edges, over the rod's `[vertices]` in pale grey."""
    camera = disk_camera(vertices)
    points = vertices.batch()
    whole = in_rod[topology.elements[2]].all(axis=-1)
    layers = [contours.picture(points, topology.subset(whole, -1), np.zeros(topology.n_vertices), rod_colour, camera),
              simplices.strokes(points, topology, mesh_colour, mesh_width, camera),
              rod_edge(vertices, topology, in_rod, camera)]
    return image_figure(on_white(simplices.composited(layers)))


def animate(topology: Topology, vertices: Extensor, in_rod: np.ndarray, penetration: float,
            frames: Iterable[tuple[Extensor, Extensor]]) -> Iterator[np.ndarray]:
    """The rod at every frame of its potential and supercurrent `[2] Scalar+[vertices]` as it comes, the
    superconductor left and the perfect conductor right: coloured by the supercurrent, with the field lines and the
    rod's edge over it."""
    camera = disk_camera(vertices)
    points = vertices.batch()
    edge = rod_edge(vertices, topology, in_rod, camera)
    lines = contours.Lines(line_spacing, line_spacing / 2, np.append(line_colour, 1.0), line_width)
    gap = np.full((camera.shape[0], gap_pixels, 3), 255, dtype=np.uint8)
    for potentials, currents in frames:
        panels = []
        for potential, current in zip(potentials.batch().to_array().reshape(2, -1),
                                      currents.batch().to_array().reshape(2, -1)):
            colours = current_map(0.5 + np.clip(current * penetration / current_share, -1, 1) / 2)[:, :3]
            panels.append(on_white(simplices.composited([
                contours.picture(points, topology, current, colours, camera),
                contours.picture(points, topology, potential, colours, camera, lines=(lines,), opacity=0.0),
                edge])))
        yield np.concatenate([panels[0], gap, panels[1]], axis=1)


@style()
def draw_penetration(depths: Extensor, strengths: Extensor, penetration: float) -> Figure:
    """The field's strength in the triangles along a radius `[triangles] Scalar+`, against their depth below the rod's
    surface `[triangles] Scalar+` in penetration depths, on a logarithmic axis, beside London's exponential decay from
    the shallowest triangle's strength."""
    figure, ax = subplots(chart_width, chart_height)
    depth_values, strength_values = depths.to_array(), strengths.to_array()
    order = np.argsort(depth_values)
    scaled = depth_values[order] / penetration
    ax.plot(scaled, strength_values[order], "o", color=measured_colour, fillstyle="none")
    ax.plot(scaled, strength_values[order][0] * np.exp(-(scaled - scaled[0])), color=london_colour)
    ax.set_yscale("log")
    ax.set_xlabel("depth / penetration depth")
    ax.set_ylabel("field / applied field")
    figure.tight_layout(pad=0.4)
    return figure


def draw_trapped(topology: Topology, vertices: Extensor, in_hole: np.ndarray, fields: Extensor) -> Figure:
    """The ring holding each trapped field `[quanta] Scalar+[vertices]` side by side, coloured by the field's strength
    along its axis, white for none, with its current's lines, the strength's level lines, and the hole's edge."""
    mv = vertices.context.multivector
    camera = Camera(mv.x, mv.y, mv.z, 0 * mv.x, 1.0, np.zeros(2), np.array([1.0, 1.0]), ring_pixels)
    points = vertices.batch()
    hole_edge = rod_edge(vertices, topology, in_hole, camera)
    lines = contours.Lines(current_spacing, current_spacing / 2, np.append(line_colour, 1.0), line_width)
    gap = np.full((camera.shape[0], gap_pixels, 3), 255, dtype=np.uint8)
    panels = []
    for field in fields.batch().to_array().reshape(len(fields.batch()), -1):
        colours = strength_map(np.clip(field / strongest, 0, 1) * 0.85)[:, :3]
        panels.append(on_white(simplices.composited([
            contours.picture(points, topology, field, colours, camera),
            contours.picture(points, topology, field, colours, camera, lines=(lines,), opacity=0.0), hole_edge])))
    return image_figure(np.concatenate([part for panel in panels for part in (panel, gap)][:-1], axis=1))


@style()
def draw_staircase(cooling_fields: np.ndarray, kept: Extensor) -> Figure:
    """The flux a ring keeps once the field it was cooled in is switched off `[fields] Scalar+`, in flux quanta,
    against that field, with the whole numbers of quanta marked."""
    figure, ax = subplots(chart_width, chart_height)
    flux = kept.to_array()
    for whole in range(int(np.ceil(flux.max())) + 1):
        ax.axhline(whole, color=mesh_colour, linewidth=0.6)
    ax.plot(cooling_fields, flux, color=measured_colour, drawstyle="steps-post")
    ax.set_yticks(range(int(np.ceil(flux.max())) + 1))
    ax.set_xlabel("cooling field")
    ax.set_ylabel("kept flux / flux quantum")
    figure.tight_layout(pad=0.4)
    return figure
