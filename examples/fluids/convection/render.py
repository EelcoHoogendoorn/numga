"""Drawing for Rayleigh-Bénard convection: the layer of triangles, the layer's temperature and streamlines over time,
and the growth of a disturbance against the Rayleigh number as a chart.

A layer closed on itself along its length is drawn opened at its seam, its first column again a period further along.
Its points are read by the planes x and y and weighed by their volume, `mv.v & point`, the algebra's own projective
weight. The temperature is red above the middle's and blue
below, the streamlines are the stream function's level lines, and the floor and the ceiling are strokes in the colours
of their temperatures.
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
stream_colour = np.array(to_rgb("#2b2f33"))
growth_colour, axis_colour = np.array(to_rgb("#1d4ed8")), np.array(to_rgb("#6b7280"))
temperature_map = colormaps["RdBu_r"]
# The layer's picture: pixels across and the room around it in depths; the widths of the walls, the mesh's edges and the
# streamlines; the streamlines' spacing in units of the stream function, the same in every frame, so that how close
# they lie shows how strongly the liquid flows.
n_pixels, margin = 320, 0.04
wall_width, mesh_width, stream_width = 2.5, 1.0, 1.0
stream_spacing = 1.5
# The chart's pixels.
chart_width, chart_height = 300, 220


# --- plumbing -------------------------------------------------------------------------
def on_white(picture: np.ndarray) -> np.ndarray:
    """Colour and coverage `[rows, columns, 4]` laid over white, as bytes `[rows, columns, 3]`."""
    shown = picture[..., :3] + (1 - picture[..., 3:])
    return np.round(np.clip(shown, 0, 1) * 255).astype(np.uint8)


def layer_camera(vertices: Extensor, period: float) -> Camera:
    """The camera looking at the layer side on, `n_pixels` across: points read by the planes x and y, weighed by their
    volume."""
    mv = vertices.context.multivector
    return Camera(mv.x, mv.y, 0 * mv.x, mv.v, 0.0, np.array([period / 2, 0.5]),
                  np.array([period / 2 + margin, 0.5 + margin]), n_pixels)


def walls(topology: Topology, points: Extensor, camera: Camera):
    """The floor and the ceiling, the edges of the open layer's border that run along it, each in the colour of its
    temperature, warm at the floor and cool at the ceiling."""
    border = topology.border
    tails, heads = border.incidence[1, 0] * points                              # Point-[border], Point-[border]
    heights = (points.context.multivector.y & tails).to_array()                 # [border]
    level = np.isclose(heights, (points.context.multivector.y & heads).to_array())   # [border]
    colours = temperature_map(1 - heights[level])[:, :3]                       # [walls, 3]
    return simplices.strokes(points, border.subset(level, 1), colours, wall_width, camera)


# --- drawing --------------------------------------------------------------------------
def draw_layer(topology: Topology, vertices: Extensor, period: float) -> Figure:
    """The open layer's triangles as strokes of their edges, between its warm floor and cool ceiling."""
    camera = layer_camera(vertices, period)
    layers = [simplices.strokes(vertices, topology, mesh_colour, mesh_width, camera), walls(topology, vertices, camera)]
    return image_figure(on_white(simplices.composited(layers)))


def animate(topology: Topology, vertices: Extensor, opening: Extensor, period: float,
            frames: Iterable[tuple[Extensor, Extensor]]) -> Iterator[np.ndarray]:
    """The open layer at every frame of the closed layer's stream function and temperature `Scalar+[vertices]` as it
    comes, read on the open layer by `opening`: coloured by its temperature, red above the middle's and blue below,
    with streamlines at a fixed spacing of the stream function, between its floor and ceiling."""
    camera = layer_camera(vertices, period)
    boundaries = [walls(topology, vertices, camera)]
    lines = contours.Lines(stream_spacing, stream_spacing / 2, np.append(stream_colour, 1.0), stream_width)
    for stream, temperature in frames:
        values = (opening * temperature).to_array()
        levels = (opening * stream).to_array()
        colours = temperature_map(np.clip(0.5 + values, 0, 1))[:, :3]
        yield on_white(simplices.composited([
            contours.picture(vertices, topology, values, colours, camera),
            contours.picture(vertices, topology, levels, colours, camera, lines=(lines,), opacity=0.0)] + boundaries))


@style()
def draw_turnover(times: np.ndarray, sizes: Extensor) -> Figure:
    """The flow's squared size over the layer `[frames] Scalar+` against time on a logarithmic scale: a straight rise
    while the rolls grow, then the swings as they overshoot and settle."""
    figure, ax = subplots(chart_width, chart_height)
    ax.semilogy(times, sizes.to_array(), color=growth_colour)
    ax.set_xlim(times[0], times[-1])
    ax.set_xticks([times[0], times[-1]], [f"{times[0]:g}", f"{times[-1]:g}"])
    ax.set_xlabel("time")
    ax.set_ylabel("flow")
    figure.tight_layout(pad=0.4)
    return figure
