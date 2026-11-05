"""Drawing for Rayleigh-Bénard convection: the layer of triangles, the layer's temperature and streamlines over time,
and as charts the flow's growth and the layer's energy budget.

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
from examples.util.charts import FRAME_WIDTH, points, style, subplots
from examples.util.wgsl import contours, image_figure, simplices
from examples.util.wgsl.camera import Camera
from examples.fluids.convection import core

mv = core.mv
Topology = simplices.Topology
mesh_colour = np.array(to_rgb("#9aa5ae"))
stream_colour = np.array(to_rgb("#2b2f33"))
growth_colour, axis_colour = np.array(to_rgb("#1d4ed8")), np.array(to_rgb("#6b7280"))
# The budget's terms: the kinetic and the potential energy, the energy dissipated, the potential energy the heat moved,
# and the entropy produced.
kinetic_colour, potential_colour = np.array(to_rgb("#1d4ed8")), np.array(to_rgb("#b91c1c"))
dissipated_colour, conducted_colour = np.array(to_rgb("#6b7280")), np.array(to_rgb("#d97706"))
entropy_colour, residual_colour = np.array(to_rgb("#15803d")), np.array(to_rgb("#2b2f33"))
temperature_map = colormaps["RdBu_r"]
# The temperature's bands, centred on evenly spaced levels from the ceiling's to the floor's, an odd count so that the
# well-mixed middle is one band, white.
n_bands = 7
# The layer's picture: pixels across and the room around it in depths; the widths of the walls, the mesh's edges and the
# streamlines; the streamlines' spacing in units of the stream function, the same in every frame, so that how close
# they lie shows how strongly the liquid flows.
n_pixels, margin = 320, 0.04
wall_width, mesh_width, stream_width = 2.5, 1.0, 1.0
stream_spacing = 3.0
# The charts' pixels, and the range of the budget's residual as a share of the energy dissipated, from below round-off
# to well above it, marked at a part in a trillion and at its top.
chart_width, chart_height, budget_height = 300, 220, 376
residual_range, residual_marks = (1e-15, 1e-10), (1e-12, 1e-10)
# The residual's dots, in pixels across.
residual_dot = 4


# --- plumbing -------------------------------------------------------------------------
def on_white(picture: np.ndarray) -> np.ndarray:
    """Colour and coverage `[rows, columns, 4]` laid over white, as bytes `[rows, columns, 3]`."""
    shown = picture[..., :3] + (1 - picture[..., 3:])
    return np.round(np.clip(shown, 0, 1) * 255).astype(np.uint8)


def layer_camera(period: float) -> Camera:
    """The camera looking at the layer side on, `n_pixels` across: points read by the planes x and y, weighed by their
    volume."""
    return Camera(mv.x, mv.y, 0 * mv.x, mv.v, 0.0, np.array([period / 2, 0.5]),
                  np.array([period / 2 + margin, 0.5 + margin]), n_pixels)


def walls(topology: Topology, points: Extensor, camera: Camera):
    """The floor and the ceiling, the edges of the open layer's border that run along it, each in the colour of its
    temperature, warm at the floor and cool at the ceiling."""
    border = topology.border
    tails, heads = border.incidence[1, 0] * points                              # Point-[border], Point-[border]
    heights = (mv.y & tails).to_array()                                         # [border]
    level = np.isclose(heights, (mv.y & heads).to_array())                      # [border]
    colours = temperature_map(1 - heights[level])[:, :3]                       # [walls, 3]
    return simplices.strokes(points, border.subset(level, 1), colours, wall_width, camera)


# --- drawing --------------------------------------------------------------------------
def draw_layer(topology: Topology, vertices: Extensor, period: float) -> Figure:
    """The open layer's triangles as strokes of their edges, between its warm floor and cool ceiling."""
    camera = layer_camera(period)
    layers = [simplices.strokes(vertices, topology, mesh_colour, mesh_width, camera), walls(topology, vertices, camera)]
    return image_figure(on_white(simplices.composited(layers)))


def animate(topology: Topology, vertices: Extensor, opening: Extensor, period: float,
            frames: Iterable[tuple[Extensor, Extensor, object]]) -> Iterator[np.ndarray]:
    """The open layer at every frame of the closed layer's stream function and temperature `Scalar+[vertices]` as it
    comes, with the budget it leaves undrawn, read on the open layer by `opening`: coloured by its temperature in
    bands, red above the middle's and blue below, with streamlines at a fixed spacing of the stream function, between
    its floor and ceiling."""
    camera = layer_camera(period)
    boundaries = [walls(topology, vertices, camera)]
    lines = contours.Lines(stream_spacing, stream_spacing / 2, np.append(stream_colour, 1.0), stream_width)
    # Each band in the colour of the temperature at its middle, flat across it, so that a frame changes only where
    # the bands' edges and the streamlines move.
    spacing = 1 / (n_bands - 1)
    middles = np.linspace(-0.5, 0.5, n_bands)
    bands = contours.Bands(-0.5 - spacing / 2, spacing, temperature_map(0.5 + middles))
    for stream, temperature, _ in frames:
        values = (opening * temperature).to_array()
        levels = (opening * stream).to_array()
        yield on_white(simplices.composited([
            contours.picture(vertices, topology, values, np.ones(3), camera, bands=(bands,)),
            contours.picture(vertices, topology, levels, np.ones(3), camera, lines=(lines,), opacity=0.0)] + boundaries))


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


@style()
def draw_budget(times: np.ndarray, kinetic: Extensor, potential: Extensor, dissipated: Extensor, conducted: Extensor,
                residual: Extensor, produced: Extensor) -> Figure:
    """The layer's energy budget against time `[frames] Scalar+` each, in three charts kept in step: above, the
    changes of the kinetic and the potential energy since the start, the energy dissipated, and minus the potential
    energy the heat moved, which add up to nothing; in the middle, what they leave unaccounted as a share of the energy
    dissipated, on a logarithmic scale; below, the entropy produced."""
    figure, (energies, unaccounted, entropy) = subplots(chart_width, budget_height, nrows=3, sharex=True,
                                                         gridspec_kw=dict(height_ratios=(2, 1, 1)))
    for values, colour, label in ((kinetic, kinetic_colour, "kinetic"), (potential, potential_colour, "potential"),
                                  (dissipated, dissipated_colour, "dissipated"),
                                  (-conducted, conducted_colour, "moved by heat")):
        energies.plot(times, values.to_array(), color=colour, label=label)
    energies.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1), ncol=2, handlelength=1.2,
                    columnspacing=1.0, labelspacing=0.2, borderaxespad=0.1)
    energies.axhline(0, color=axis_colour, linewidth=FRAME_WIDTH)
    energies.set_yticks([])
    energies.set_ylabel("energy")
    unaccounted.semilogy(times, np.abs(residual.to_array()), "o", color=residual_colour,
                         markersize=points(residual_dot), markeredgewidth=0)
    unaccounted.set_ylim(*residual_range)
    unaccounted.set_yticks(residual_marks, [f"$10^{{{np.log10(mark):.0f}}}$" for mark in residual_marks])
    unaccounted.minorticks_off()
    unaccounted.set_ylabel("residual")
    entropy.plot(times, produced.to_array(), color=entropy_colour)
    entropy.set_ylim(0, None)
    entropy.set_yticks([])
    entropy.set_ylabel("entropy")
    entropy.set_xlim(times[0], times[-1])
    entropy.set_xticks([times[0], times[-1]], [f"{times[0]:g}", f"{times[-1]:g}"])
    entropy.set_xlabel("time")
    for ax in (energies, unaccounted, entropy):
        ax.spines[["top", "right"]].set_visible(False)
    energies.spines["bottom"].set_visible(False)
    figure.align_ylabels()
    figure.tight_layout(pad=0.4, h_pad=0.3)
    return figure
