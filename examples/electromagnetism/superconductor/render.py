"""Rendering for Meissner expulsion in a superconducting rod and flux quantization in a superconducting ring.

Cross-section points are read by the `x` and `y` planes at unit weight. Level contours of the axial potential trace
magnetic field lines across the rod, red and blue shading marks supercurrent flowing out of and into the cross-section,
and purple shading with current contours depicts quantized axial magnetic fields trapped in a superconducting ring.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure

from examples.util.charts import style, subplots
from examples.util.panels import image_figure
from numga.extensor import Extensor
from numga.render import Camera, contours, drawing, simplices
from numga.topology import Topology

grid_colour = np.array(to_rgb("#9aa5ae"))
rod_colour = np.array(to_rgb("#e3e0da"))
edge_colour = np.array(to_rgb("#7d7466"))
line_colour = np.array(to_rgb("#2b2f33"))
current_map, strength_map = colormaps["RdBu_r"], colormaps["Purples"]
measured_colour, london_colour = np.array(to_rgb("#1d4ed8")), np.array(to_rgb("#c2410c"))

chart_width, chart_height = 300, 220
ring_pixels, current_spacing, strongest_field = 130, 0.25, 3.0
n_pixels = 200
grid_width, edge_width, line_width = 0.75, 1.5, 1.2
line_spacing = 0.2
current_share = 1.5
gap_pixels = 6


# --- plumbing -------------------------------------------------------------------------
def white_image(picture: np.ndarray) -> np.ndarray:
    """Alpha compositing over a white background converts RGBA floats (`[..., 4]`) into uint8 RGB bytes (`[..., 3]`)."""
    rgb_image = picture[..., :3] + (1 - picture[..., 3:])
    return np.round(np.clip(rgb_image, 0, 1) * 255).astype(np.uint8)


def disk_camera(vertices: Extensor) -> Camera:
    """Orthographic pairing with the `x` and `y` planes frames the cross-section disk at `n_pixels` across."""
    mv = vertices.context.multivector
    half_extent = float((mv.x & vertices.batch()).abs().to_array().max())
    return Camera(mv.x, mv.y, mv.z, 0 * mv.x, 1.0, np.zeros(2), np.array([half_extent, half_extent]), n_pixels)


def region_border(vertices: Extensor, topology: Topology, region_mask: np.ndarray, camera: Camera) -> drawing.Draw:
    """Border edges of triangles contained in `region_mask` (`[V]`) form the region outline strokes."""
    region_faces = region_mask[topology.elements[2]].all(axis=-1)
    return simplices.strokes(vertices.batch(), topology.subset(region_faces, -1).border, edge_colour, edge_width, camera)


# --- drawing --------------------------------------------------------------------------
def rod_figure(topology: Topology, vertices: Extensor, in_rod: np.ndarray) -> Figure:
    """Pale grey shading and triangle strokes depict the cross-section disk around the superconducting rod."""
    V, E, F = topology.n_elements
    camera = disk_camera(vertices)
    points = vertices.batch()
    rod_faces = in_rod[topology.elements[2]].all(axis=-1)
    layers = [
        contours.picture(points, topology.subset(rod_faces, -1), np.zeros(V), rod_colour, camera),
        simplices.strokes(points, topology, grid_colour, grid_width, camera),
        region_border(vertices, topology, in_rod, camera),
    ]
    return image_figure(white_image(drawing.composited(layers)))


def cooling_animation(
    topology: Topology,
    vertices: Extensor,
    in_rod: np.ndarray,
    penetration_depth: float,
    states: Iterable[tuple[Extensor, Extensor]],
) -> Iterator[np.ndarray]:
    """Yields side-by-side RGB frames of the superconductor and the perfect conductor (`[2] Scalar+[V]`)."""
    camera = disk_camera(vertices)
    points = vertices.batch()
    border = region_border(vertices, topology, in_rod, camera)
    field_lines = contours.Lines(line_spacing, line_spacing / 2, np.append(line_colour, 1.0), line_width)
    gap = np.full((camera.shape[0], gap_pixels, 3), 255, dtype=np.uint8)
    for potentials, currents in states:
        panels = []
        for potential, current in zip(
            potentials.batch().to_array().reshape(2, -1),
            currents.batch().to_array().reshape(2, -1),
        ):
            colours = current_map(0.5 + np.clip(current * penetration_depth / current_share, -1, 1) / 2)[:, :3]
            panels.append(white_image(drawing.composited([
                contours.picture(points, topology, current, colours, camera),
                contours.picture(points, topology, potential, colours, camera, lines=(field_lines,), opacity=0.0),
                border,
            ])))
        yield np.concatenate([panels[0], gap, panels[1]], axis=1)


@style()
def penetration_figure(depths: Extensor, strengths: Extensor, penetration_depth: float) -> Figure:
    """Logarithmic field strength (`[...] Scalar+`) against depth compares simulated screening with London decay."""
    figure, ax = subplots(chart_width, chart_height)
    order = depths.argsort()
    scaled_depths = (depths[order] / penetration_depth).to_array()
    ordered_strengths = strengths[order].to_array()
    ax.plot(scaled_depths, ordered_strengths, "o", color=measured_colour, fillstyle="none")
    ax.plot(scaled_depths, ordered_strengths[0] * np.exp(-(scaled_depths - scaled_depths[0])), color=london_colour)
    ax.set_yscale("log")
    ax.set_xlabel("depth / penetration depth")
    ax.set_ylabel("field / applied field")
    figure.tight_layout(pad=0.4)
    return figure


def trapped_figure(topology: Topology, vertices: Extensor, in_hole: np.ndarray, fields: Extensor) -> Figure:
    """Side-by-side ring panels (`[...] Scalar+[V]`) display purple axial field strength beneath current contours."""
    mv = vertices.context.multivector
    camera = Camera(mv.x, mv.y, mv.z, 0 * mv.x, 1.0, np.zeros(2), np.array([1.0, 1.0]), ring_pixels)
    points = vertices.batch()
    hole_border = region_border(vertices, topology, in_hole, camera)
    current_lines = contours.Lines(current_spacing, current_spacing / 2, np.append(line_colour, 1.0), line_width)
    gap = np.full((camera.shape[0], gap_pixels, 3), 255, dtype=np.uint8)
    panels = []
    field_batches = fields.batch()
    for field in field_batches.to_array().reshape(len(field_batches), -1):
        colours = strength_map(np.clip(field / strongest_field, 0, 1) * 0.85)[:, :3]
        panels.append(white_image(drawing.composited([
            contours.picture(points, topology, field, colours, camera),
            contours.picture(points, topology, field, colours, camera, lines=(current_lines,), opacity=0.0),
            hole_border,
        ])))
    return image_figure(np.concatenate([part for panel in panels for part in (panel, gap)][:-1], axis=1))


@style()
def staircase_figure(cooling_fields: np.ndarray, retained_flux: Extensor) -> Figure:
    """Stepwise curves show retained ring flux (`[...] Scalar+`) in flux quanta across cooling fields (`[...]`)."""
    figure, ax = subplots(chart_width, chart_height)
    flux_values = retained_flux.to_array()
    n_levels = int(np.ceil(flux_values.max())) + 1
    for level in range(n_levels):
        ax.axhline(level, color=grid_colour, linewidth=0.6)
    ax.plot(cooling_fields, flux_values, color=measured_colour, drawstyle="steps-post")
    ax.set_yticks(range(n_levels))
    ax.set_xlabel("cooling field")
    ax.set_ylabel("kept flux / flux quantum")
    figure.tight_layout(pad=0.4)
    return figure
