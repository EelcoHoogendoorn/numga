"""Drawing for Rayleigh-Bénard convection: the layer of triangles, the layer's temperature and streamlines over time,
and the growth of a disturbance against the Rayleigh number as a chart.

A layer closed on itself along its length is opened at its seam to be drawn: its first column again a period further
along, and the triangles across the seam joined to that copy. Its points are read by the planes x and y and weighed by
their volume, `mv.v & point`, the algebra's own projective weight. The temperature is red above the middle's and blue
below, the streamlines are the stream function's level lines, and the floor and the ceiling are strokes in the colours
of their temperatures.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure

from numga import concatenate
from numga.extensor import Extensor
from examples.util.charts import style, subplots
from examples.util.wgsl import contours, image_figure, simplices
from examples.util.wgsl.camera import Camera

Topology = simplices.Topology
mesh_colour = np.array(to_rgb("#9aa5ae"))
stream_colour = np.array(to_rgb("#2b2f33"))
growth_colour, axis_colour = np.array(to_rgb("#1d4ed8")), np.array(to_rgb("#6b7280"))
temperature_map = colormaps["RdBu_r"]
# The layer's picture: pixels across and the room around it in depths; the widths of the walls, the mesh's edges and the
# streamlines; the streamlines' spacing in units of the stream function, a share of its largest value in each frame.
n_pixels, margin = 320, 0.04
wall_width, mesh_width, stream_width = 2.5, 0.75, 1.0
n_streamlines = 8
# The chart's pixels, and the onset of convection between rigid walls marked on it.
chart_width, chart_height = 300, 220
critical_rayleigh = 1708


# --- plumbing -------------------------------------------------------------------------
def on_white(picture: np.ndarray) -> np.ndarray:
    """Colour and coverage `[rows, columns, 4]` laid over white, as bytes `[rows, columns, 3]`."""
    shown = picture[..., :3] + (1 - picture[..., 3:])
    return np.round(np.clip(shown, 0, 1) * 255).astype(np.uint8)


def unrolled(topology: Topology, vertices: Extensor, n_across: int, period: float) -> tuple[Extensor, Topology]:
    """A layer closed on itself, opened at its seam for drawing: its vertices `Point-[vertices]` with its first column
    again a period further along, and its triangles, those across the seam joined to that copy."""
    faces = topology.elements[2]
    n_vertices = topology.n_vertices
    columns = faces // n_across
    across_seam = (columns.max(axis=-1) - columns.min(axis=-1) > 1)[:, None] & (columns == 0)
    faces = np.where(across_seam, n_vertices + faces % n_across, faces)
    points = vertices.batch()                                                  # [vertices] Point-
    first = points[:n_across] + vertices.context.multivector.x.dual() * period  # [across] Point-
    return concatenate([points, first]), Topology(n_vertices + n_across, faces)


def closing(values: np.ndarray, n_across: int) -> np.ndarray:
    """Values at a layer's vertices `[vertices]`, with its first column's again for the copy at the seam."""
    return np.concatenate([values, values[:n_across]])


def layer_camera(vertices: Extensor, period: float) -> Camera:
    """The camera looking at the layer side on, `n_pixels` across: points read by the planes x and y, weighed by their
    volume."""
    mv = vertices.context.multivector
    return Camera(mv.x, mv.y, 0 * mv.x, mv.v, 0.0, np.array([period / 2, 0.5]),
                  np.array([period / 2 + margin, 0.5 + margin]), n_pixels)


def walls(points: Extensor, n_across: int, camera: Camera):
    """The floor and the ceiling, opened at the seam `[columns + 1, across] Point-`, in the colours of their
    temperatures."""
    rows = points.reshape(-1, n_across)
    n_points = rows.shape[0]
    colours = np.concatenate([np.repeat(np.array(temperature_map(1.0))[None, :3], n_points - 1, axis=0),
                              np.repeat(np.array(temperature_map(0.0))[None, :3], n_points - 1, axis=0)])
    return simplices.strokes(concatenate([rows[:, 0], rows[:, -1]]), Topology.chains([n_points, n_points]), colours,
                             wall_width, camera)


# --- drawing --------------------------------------------------------------------------
def draw_layer(topology: Topology, vertices: Extensor, n_across: int, period: float) -> Figure:
    """The layer's triangles as strokes of their edges, between its warm floor and cool ceiling."""
    points, faces = unrolled(topology, vertices, n_across, period)
    camera = layer_camera(vertices, period)
    layers = [simplices.strokes(points, faces, mesh_colour, mesh_width, camera), walls(points, n_across, camera)]
    return image_figure(on_white(simplices.composited(layers)))


def animate(topology: Topology, vertices: Extensor, n_across: int, period: float,
            frames: Iterable[tuple[Extensor, Extensor]]) -> Iterator[np.ndarray]:
    """The layer at every frame of its stream function and temperature `Scalar+[vertices]` as it comes: coloured by its
    temperature, red above the middle's and blue below, with streamlines spaced by a share of the frame's strongest
    flow, between its floor and ceiling."""
    points, faces = unrolled(topology, vertices, n_across, period)
    camera = layer_camera(vertices, period)
    over = [walls(points, n_across, camera)]
    for stream, temperature in frames:
        values = closing(temperature.to_array(), n_across)
        levels = closing(stream.to_array(), n_across)
        colours = temperature_map(np.clip(0.5 + values, 0, 1))[:, :3]
        spacing = max(np.abs(levels).max(), 1e-12) / n_streamlines
        lines = contours.Lines(spacing, spacing / 2, np.append(stream_colour, 1.0), stream_width)
        yield on_white(simplices.composited([
            contours.picture(points, faces, values, colours, camera),
            contours.picture(points, faces, levels, colours, camera, lines=(lines,), opacity=0.0)] + over))


@style()
def draw_turnover(times: np.ndarray, sizes: Extensor, shown: float) -> Figure:
    """The flow's squared size over the layer `[frames] Scalar+` against time on a logarithmic scale, from the moment the
    animation starts marked: a straight rise while the rolls grow, then the swings as they overshoot and settle."""
    figure, ax = subplots(chart_width, chart_height)
    ax.axvline(shown, color=axis_colour, linewidth=0.8, linestyle=":")
    ax.semilogy(times, sizes.to_array(), color=growth_colour)
    ax.set_xlim(times[0], times[-1])
    ax.set_xticks([times[0], times[-1]], [f"{times[0]:g}", f"{times[-1]:g}"])
    ax.set_xlabel("time")
    ax.set_ylabel("flow")
    figure.tight_layout(pad=0.4)
    return figure


@style()
def draw_onset(rayleighs: np.ndarray, growth: Extensor) -> Figure:
    """The growth rate of a small disturbance `[rayleighs] Scalar+` against the Rayleigh number, with its zero, and the
    onset between rigid walls marked."""
    figure, ax = subplots(chart_width, chart_height)
    ax.axhline(0, color=axis_colour, linewidth=0.8)
    ax.axvline(critical_rayleigh, color=axis_colour, linewidth=0.8, linestyle=":")
    ax.plot(rayleighs, growth.to_array(), "o-", color=growth_colour)
    ax.set_xticks([rayleighs[0], critical_rayleigh, rayleighs[-1]])
    ax.set_xlabel("Rayleigh number")
    ax.set_ylabel("growth rate")
    figure.tight_layout(pad=0.4)
    return figure
