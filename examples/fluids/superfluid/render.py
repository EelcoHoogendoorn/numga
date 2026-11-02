"""Drawing for superfluid helium: its sounds against temperature as a chart, and the cavity with its wall of powder,
as a mesh and as its temperature and pressure over time.

The cavity's points are read by the planes x and y and weighed by their mass, `mv.m & point`, the algebra's own
projective weight. A rise of temperature is drawn warm red and a fall cold blue, a rise of pressure orange and a fall
purple; the powder is a pale tan under the mesh, and its edges grey strokes over the fields.
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
wall_colour = np.array(to_rgb("#3b4a57"))
mesh_colour = np.array(to_rgb("#9aa5ae"))
powder_colour = np.array(to_rgb("#e8d9b5"))
edge_colour = np.array(to_rgb("#7d7466"))
temperature_map, pressure_map = colormaps["RdBu_r"], colormaps["PuOr_r"]
second_colour, fourth_colour, first_colour = (np.array(to_rgb(colour)) for colour in ("#c2410c", "#1d4ed8", "#6b7280"))
# The chart's pixels.
chart_width, chart_height = 320, 240
# The cavity's picture: pixels across and the room around it in millimetres; the widths of the walls, the mesh's edges
# and the powder's edge; the pixels between the two pictures of the cavity.
n_pixels, margin = 320, 0.3
wall_width, mesh_width, edge_width = 2.0, 0.75, 1.5
gap_pixels = 6


# --- plumbing -------------------------------------------------------------------------
def on_white(picture: np.ndarray) -> np.ndarray:
    """Colour and coverage `[rows, columns, 4]` laid over white, as bytes `[rows, columns, 3]`."""
    shown = picture[..., :3] + (1 - picture[..., 3:])
    return np.round(np.clip(shown, 0, 1) * 255).astype(np.uint8)


def readings(plane: Extensor, values: Extensor) -> np.ndarray:
    """Values read by a plane, as numbers in their field's shape."""
    return (plane & values.batch()).to_array().reshape(values.batch().shape)


def cavity_camera(vertices: Extensor) -> Camera:
    """The camera looking down on the cavity, `n_pixels` across: points read by the planes x and y, weighed by their
    mass."""
    mv = vertices.context.multivector
    across, up = readings(mv.x, vertices), readings(mv.y, vertices)
    middle = np.array([across.max() + across.min(), up.max() + up.min()]) / 2
    half = np.array([across.max() - across.min(), up.max() - up.min()]) / 2 + margin
    return Camera(mv.x, mv.y, 0 * mv.x, mv.m, 0.0, middle, half, n_pixels)


def walls(vertices: Extensor, topology: Topology, camera: Camera):
    """The cavity's walls as strokes."""
    return simplices.strokes(vertices.batch(), topology.border, wall_colour, wall_width, camera)


def powder_edge(vertices: Extensor, topology: Topology, in_powder: np.ndarray, camera: Camera):
    """The border of the powder's triangles `[faces]` as strokes, the walls drawn over it leaving its edge into the open
    liquid."""
    return simplices.strokes(vertices.batch(), topology.subset(in_powder).border, edge_colour, edge_width, camera)


# --- drawing --------------------------------------------------------------------------
@style()
def draw_sounds(temperatures: np.ndarray, free: Extensor, held: Extensor, first_sounds: np.ndarray,
                second_sounds: np.ndarray, fourth_sounds: np.ndarray) -> Figure:
    """The speeds of sound over temperature, from the squared speeds of the liquid `[temperatures, 2] Scalar+` and of
    the liquid with its normal fluid held `[temperatures, 2] Scalar+` as curves, the measured second and fourth sound as
    points, and the first sound the liquid was given as a faint curve, on a logarithmic axis of speed."""
    figure, ax = subplots(chart_width, chart_height)
    curves = ((first_sounds, None, first_colour, "first"),
              (np.sqrt(np.abs(held.to_array()[:, 1])), fourth_sounds, fourth_colour, "fourth"),
              (np.sqrt(free.to_array()[:, 0]), second_sounds, second_colour, "second"))
    for computed, measured, colour, name in curves:
        ax.plot(temperatures, computed, color=colour)
        if measured is not None:
            ax.plot(temperatures, measured, "o", color=colour, fillstyle="none")
        # Each curve named where it starts, above it.
        ax.annotate(name, (temperatures[0], computed[0]), xytext=(2, 5), textcoords="offset pixels", color=colour)
    ax.set_yscale("log")
    ax.set_yticks([20, 50, 100, 200])
    ax.set_yticklabels(["20", "50", "100", "200"])
    ax.minorticks_off()
    ax.set_xlabel("temperature (K)")
    ax.set_ylabel("speed (m/s)")
    figure.tight_layout(pad=0.4)
    return figure


def draw_cavity(topology: Topology, vertices: Extensor, in_powder: np.ndarray) -> Figure:
    """The cavity's triangles as strokes of their edges, over the powder `[faces]` in pale tan, within its walls."""
    camera = cavity_camera(vertices)
    points = vertices.batch()
    layers = [contours.picture(points, topology.subset(in_powder), np.zeros(topology.n_vertices), powder_colour,
                               camera),
              simplices.strokes(points, topology, mesh_colour, mesh_width, camera),
              walls(vertices, topology, camera)]
    return image_figure(on_white(simplices.composited(layers)))


# The flat bands each rise is drawn in, from falling fully to rising fully: an odd count, so that one band holds no rise.
n_bands = 9
band_spacing = 2 / (n_bands - 1)
band_middles = np.linspace(-1, 1, n_bands)


def animate(topology: Topology, vertices: Extensor, in_powder: np.ndarray, frames: Iterable[tuple[Extensor, Extensor]],
            warmest: float, highest: float) -> Iterator[np.ndarray]:
    """The cavity twice at every frame as it comes, coloured above by its rise of temperature and below by its rise of
    pressure `Scalar+[vertices]` each, in flat bands, saturating at these rises of temperature and of pressure, with
    the powder's `[faces]` edges and the walls drawn over them."""
    camera = cavity_camera(vertices)
    points = vertices.batch()
    over = [powder_edge(vertices, topology, in_powder, camera), walls(vertices, topology, camera)]
    gap = np.full((gap_pixels, camera.shape[1], 3), 255, dtype=np.uint8)
    for warmth, pressure in frames:
        pictures = []
        for field, scale, shades in ((warmth, warmest, temperature_map), (pressure, highest, pressure_map)):
            shares = np.clip(field.batch().to_array() / scale, -1, 1)
            bands = contours.Bands(-1 - band_spacing / 2, band_spacing, shades(0.5 + band_middles / 2))
            pictures.append(on_white(simplices.composited(
                [contours.picture(points, topology, shares, shades(0.5 + shares / 2)[:, :3], camera,
                                  bands=(bands,))] + over)))
        yield np.concatenate([pictures[0], gap, pictures[1]])


@style()
def draw_fountain(times: np.ndarray, pressures: Extensor, fountain: Extensor) -> Figure:
    """The difference of mean pressure between the heated chamber and the other `[frames] Scalar+` over time in
    milliseconds, sloshing through the powder, and the fountain pressure of their difference of mean temperature
    `[frames] Scalar+` it sloshes about, both in pascals."""
    figure, ax = subplots(chart_width, chart_height)
    ax.axhline(0, color=first_colour, linewidth=0.8)
    ax.plot(times, pressures.to_array(), color=fourth_colour)
    ax.plot(times, fountain.to_array(), color=second_colour)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("pressure (Pa)")
    figure.tight_layout(pad=0.4)
    return figure
