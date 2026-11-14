"""This module renders sound-speed and fountain-pressure charts together with triangulated cavity views and temperature-pressure animations for superfluid helium.

Cavity vertices are projected onto the horizontal and vertical coordinate planes and normalized by their projective weight. Positive and negative temperature perturbations map to warm red and cool blue bands, pressure perturbations map to orange and purple bands, and the porous powder wall appears as pale tan fill beneath triangle strokes or grey border strokes over the fields.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure

from numga import stack
from numga.render import Camera, contours, drawing, simplices
from numga.topology import Topology
from examples.fluids.superfluid import core
from examples.util.charts import style, subplots
from examples.util.panels import image_figure

mv = core.mv
wall_colour = np.array(to_rgb("#3b4a57"))
wire_colour = np.array(to_rgb("#9aa5ae"))
powder_colour = np.array(to_rgb("#e8d9b5"))
edge_colour = np.array(to_rgb("#7d7466"))
temperature_map, pressure_map = colormaps["RdBu_r"], colormaps["PuOr_r"]
second_colour, fourth_colour, first_colour = (np.array(to_rgb(colour)) for colour in ("#c2410c", "#1d4ed8", "#6b7280"))
chart_width, chart_height = 320, 240
n_pixels, margin = 320, 0.3
wall_width, wire_width, edge_width = 2.0, 0.75, 1.5
gap_pixels = 6


# --- plumbing -------------------------------------------------------------------------
def white_composite(picture: np.ndarray) -> np.ndarray:
    """Alpha compositing over a white background converts premultiplied RGBA (`[H, W, 4]`) into uint8 RGB (`[H, W, 3]`)."""
    shown = picture[..., :3] + (1 - picture[..., 3:])
    return np.round(np.clip(shown, 0, 1) * 255).astype(np.uint8)


def cavity_camera(vertices: core.Point) -> Camera:
    """An orthographic camera frames the cavity vertices (`Point-[V]`) weighed by the projective coordinate plane."""
    points = vertices.batch()
    coordinates = (stack((mv.x, mv.y)) & (points / (mv.w & points))[..., None]).to_array()
    middle = (coordinates.max(axis=0) + coordinates.min(axis=0)) / 2
    half = (coordinates.max(axis=0) - coordinates.min(axis=0)) / 2 + margin
    return Camera(mv.x, mv.y, 0 * mv.x, mv.w, 0.0, middle, half, n_pixels)


def wall_strokes(vertices: core.Point, topology: Topology, camera: Camera) -> np.ndarray:
    """Boundary strokes mark the outer walls of the triangulated cavity."""
    return simplices.strokes(vertices.batch(), topology.border, wall_colour, wall_width, camera)


def powder_border(vertices: core.Point, topology: Topology, powder_triangles: np.ndarray, camera: Camera) -> np.ndarray:
    """Border strokes outline the porous-wall triangles (`[F]`) where they meet the open chambers."""
    return simplices.strokes(vertices.batch(), topology.subset(powder_triangles, 2).border, edge_colour, edge_width, camera)


# --- drawing --------------------------------------------------------------------------
@style()
def sounds_figure(
    temperatures: np.ndarray,
    bulk_sounds: core.Scalar,
    powder_sounds: core.Scalar,
    first_sounds: np.ndarray,
    second_sounds: np.ndarray,
    fourth_sounds: np.ndarray,
) -> Figure:
    """Logarithmic speed curves compare the computed squared sound speeds (`[n_temperatures, 2] Scalar+`) in open liquid and clamped liquid against measured first, second, and fourth sound speeds."""
    figure, ax = subplots(chart_width, chart_height)
    curves = (
        (first_sounds, None, first_colour, "first"),
        (np.sqrt(np.abs(powder_sounds.to_array()[:, 1])), fourth_sounds, fourth_colour, "fourth"),
        (np.sqrt(bulk_sounds.to_array()[:, 0]), second_sounds, second_colour, "second"),
    )
    for computed, measured, colour, name in curves:
        ax.plot(temperatures, computed, color=colour)
        if measured is not None:
            ax.plot(temperatures, measured, "o", color=colour, fillstyle="none")
        ax.annotate(name, (temperatures[0], computed[0]), xytext=(2, 5), textcoords="offset pixels", color=colour)
    ax.set_yscale("log")
    ax.set_yticks([20, 50, 100, 200])
    ax.set_yticklabels(["20", "50", "100", "200"])
    ax.minorticks_off()
    ax.set_xlabel("temperature (K)")
    ax.set_ylabel("speed (m/s)")
    figure.tight_layout(pad=0.4)
    return figure


def cavity_figure(topology: Topology, vertices: core.Point, powder_triangles: np.ndarray) -> Figure:
    """Triangle edge strokes and outer wall strokes overlay the pale-tan porous wall (`[F]`) of the cavity."""
    V, E, F = topology.n_elements
    camera = cavity_camera(vertices)
    points = vertices.batch()
    layers = [
        contours.picture(points, topology.subset(powder_triangles, 2), np.zeros(V), powder_colour, camera),
        simplices.strokes(points, topology, wire_colour, wire_width, camera),
        wall_strokes(vertices, topology, camera),
    ]
    return image_figure(white_composite(drawing.composited(layers)))


# Odd band count keeps a neutral central band around zero perturbation.
n_bands = 9
band_spacing = 2 / (n_bands - 1)
band_middles = np.linspace(-1, 1, n_bands)


def cavity_animation(
    topology: Topology,
    vertices: core.Point,
    powder_triangles: np.ndarray,
    thermometer: core.Potential,
    barometer: core.Potential,
    temperature_scale: float,
    pressure_scale: float,
    states: Iterable[core.State],
) -> Iterator[np.ndarray]:
    """Yields stacked RGB frames pairing each cavity state (`State-[V]`) against `thermometer` above and `barometer` below in quantized colour bands."""
    camera = cavity_camera(vertices)
    points = vertices.batch()
    overlays = [powder_border(vertices, topology, powder_triangles, camera), wall_strokes(vertices, topology, camera)]
    gap = np.full((gap_pixels, camera.shape[1], 3), 255, dtype=np.uint8)
    panels = ((thermometer, temperature_scale, temperature_map), (barometer, pressure_scale, pressure_map))
    for state in states:
        batch_state = state.batch()
        pictures = []
        for plane, scale, shades in panels:
            shares = np.clip((plane & batch_state).to_array() / scale, -1, 1)
            bands = contours.Bands(-1 - band_spacing / 2, band_spacing, shades(0.5 + band_middles / 2))
            base = contours.picture(points, topology, shares, shades(0.5 + shares / 2)[:, :3], camera, bands=(bands,))
            pictures.append(white_composite(drawing.composited([base] + overlays)))
        yield np.concatenate([pictures[0], gap, pictures[1]])


@style()
def fountain_figure(times: np.ndarray, pressures: core.Scalar, fountain: core.Scalar) -> Figure:
    """Time-series curves compare the mean chamber pressure difference (`[n_frames] Scalar+`) against London's fountain pressure (`[n_frames] Scalar+`) in pascals."""
    figure, ax = subplots(chart_width, chart_height)
    ax.axhline(0, color=first_colour, linewidth=0.8)
    ax.plot(times, pressures.to_array(), color=fourth_colour)
    ax.plot(times, fountain.to_array(), color=second_colour)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("pressure (Pa)")
    figure.tight_layout(pad=0.4)
    return figure
