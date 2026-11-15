"""Rendering for Rayleigh-Bénard convection draws the triangulated layer, its temperature and streamlines over time, and
charts of the flow's growth and energy budget.

A periodic layer closed along its length is drawn opened at its seam one period further along. Its vertices are points
of unit `w` weight, read by pairing with `x` and `y` at `w` weight in `Camera`. Temperature is red above the mean and
blue below, streamlines are level lines of the stream function, and the floor and ceiling are drawn in the colours of
their temperatures.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure

from numga.extensor import Extensor
from examples.util.charts import FRAME_WIDTH, points, style, subplots
from numga.topology import Topology
from numga.render import Camera, contours, drawing, simplices
from examples.util.panels import image_figure
from examples.fluids.convection import core

mv = core.mv
edge_colour = np.array(to_rgb("#9aa5ae"))
stream_colour = np.array(to_rgb("#2b2f33"))
growth_colour, axis_colour = np.array(to_rgb("#1d4ed8")), np.array(to_rgb("#6b7280"))
# Budget curve colours: kinetic energy, potential energy, viscous dissipation, conductive potential-energy transfer,
# entropy production, and conservation residual.
kinetic_colour, potential_colour = np.array(to_rgb("#1d4ed8")), np.array(to_rgb("#b91c1c"))
dissipated_colour, conducted_colour = np.array(to_rgb("#6b7280")), np.array(to_rgb("#d97706"))
entropy_colour, residual_colour = np.array(to_rgb("#15803d")), np.array(to_rgb("#2b2f33"))
temperature_map = colormaps["RdBu_r"]
# Odd number of evenly spaced temperature bands between ceiling and floor so the well-mixed interior sits in the white band.
n_bands = 15
# Layer image width in pixels, margin in layer depths, stroke widths in pixels, and streamline contour spacing.
n_pixels, margin = 320, 0.04
wall_width, edge_width, stream_width = 2.5, 1.0, 1.0
stream_spacing = 3.0
# Chart pixel sizes and logarithmic range and tick marks for the normalized budget residual.
chart_width, chart_height, budget_height = 300, 220, 376
residual_range, residual_marks = (1e-15, 1e-10), (1e-12, 1e-10)
residual_dot = 4


# --- plumbing -------------------------------------------------------------------------
def on_white(picture: np.ndarray) -> np.ndarray:
    """RGBA coverage composited over a white background as RGB bytes."""
    shown = picture[..., :3] + (1 - picture[..., 3:])
    return np.round(np.clip(shown, 0, 1) * 255).astype(np.uint8)


def layer_camera(period: float) -> Camera:
    """Side-on camera across the layer: points are read by pairing with `x` and `y` at `w` weight."""
    return Camera(mv.x, mv.y, 0 * mv.x, mv.w, 0.0, np.array([period / 2, 0.5]),
                  np.array([period / 2 + margin, 0.5 + margin]), n_pixels)


def walls(topology: Topology, points: Extensor, camera: Camera):
    """Horizontal border edges at the floor and ceiling, coloured warm at the floor and cool at the ceiling."""
    border = topology.border
    tails, heads = border.incidence[1, 0] * points                              # Antivector-[E_border], Antivector-[E_border]
    heights = (mv.y & tails).to_array()                                         # [E_border]
    level = np.isclose(heights, (mv.y & heads).to_array())                      # [E_border]
    colours = temperature_map(1 - heights[level])[:, :3]                       # [E_walls, 3]
    return simplices.strokes(points, border.subset(level, 1), colours, wall_width, camera)


# --- drawing --------------------------------------------------------------------------
def draw_layer(topology: Topology, vertices: Extensor, period: float) -> Figure:
    """Triangle edges of the opened strip between its warm floor and cool ceiling."""
    camera = layer_camera(period)
    layers = [simplices.strokes(vertices, topology, edge_colour, edge_width, camera), walls(topology, vertices, camera)]
    return image_figure(on_white(drawing.composited(layers)))


def animate(topology: Topology, vertices: Extensor, opening: Extensor, period: float,
            frames: Iterable[tuple[Extensor, Extensor, object]]) -> Iterator[np.ndarray]:
    """Frames of the opened strip mapped from the closed strip's stream function and temperature:
    banded temperature from cool blue to warm red with fixed-spacing streamlines between floor and ceiling."""
    camera = layer_camera(period)
    boundaries = [walls(topology, vertices, camera)]
    lines = contours.Lines(stream_spacing, stream_spacing / 2, np.append(stream_colour, 1.0), stream_width)
    # Flat colour within each band so pixels change only where band edges and streamlines move.
    spacing = 1 / (n_bands - 1)
    middles = np.linspace(-0.5, 0.5, n_bands)
    bands = contours.Bands(-0.5 - spacing / 2, spacing, temperature_map(0.5 + middles))
    for stream, temperature, _ in frames:
        values = (opening * temperature).to_array()
        levels = (opening * stream).to_array()
        yield on_white(drawing.composited([
            contours.picture(vertices, topology, values, np.ones(3), camera, bands=(bands,)),
            contours.picture(vertices, topology, levels, np.ones(3), camera, lines=(lines,), opacity=0.0)] + boundaries))


@style()
def draw_turnover(times: np.ndarray, sizes: Extensor) -> Figure:
    """Integrated squared stream function against time on a logarithmic scale."""
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
    """Three-panel energy and entropy budget against time: changes in kinetic and potential energy
    with viscous dissipation and minus conductive potential-energy transfer above, the normalized conservation residual
    on a logarithmic scale in the middle, and cumulative entropy production below."""
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
