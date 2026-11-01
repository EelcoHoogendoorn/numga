"""Drawing for salt fingers: the tank's temperature and salinity, at the start and over time, and the growth of a
disturbance against the density ratio as a chart.

A tank closed on itself along its length is opened at its seam to be drawn: its first column again a period further
along, and the triangles across the seam joined to that copy. Its points are read by the planes x and y and weighed by
their volume, `mv.v & point`, the algebra's own projective weight. The temperature is drawn above the salinity, the same
tank in step: the temperature red above the middle's and blue below, the salinity brown above the middle's and green
below, with the walls strokes in the colours of what they hold.
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
growth_colour, axis_colour = np.array(to_rgb("#1d4ed8")), np.array(to_rgb("#6b7280"))
temperature_map, salinity_map = colormaps["RdBu_r"], colormaps["BrBG_r"]
# The tank's picture: pixels across, the room around it in depths, the rows between the two fields, and the walls'
# width; the share of each colour map the fields span, short of its darkest ends.
n_pixels, margin, n_gap = 320, 0.03, 6
wall_width = 2.5
span = 0.8
# The chart's pixels.
chart_width, chart_height = 300, 220


# --- plumbing -------------------------------------------------------------------------
def on_white(picture: np.ndarray) -> np.ndarray:
    """Colour and coverage `[rows, columns, 4]` laid over white, as bytes `[rows, columns, 3]`."""
    shown = picture[..., :3] + (1 - picture[..., 3:])
    return np.round(np.clip(shown, 0, 1) * 255).astype(np.uint8)


def unrolled(topology: Topology, vertices: Extensor, n_across: int, period: float) -> tuple[Extensor, Topology]:
    """A tank closed on itself, opened at its seam for drawing: its vertices `Point-[vertices]` with its first column
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
    """Values at a tank's vertices `[vertices]`, with its first column's again for the copy at the seam."""
    return np.concatenate([values, values[:n_across]])


def tank_camera(vertices: Extensor, period: float) -> Camera:
    """The camera looking at the tank side on, `n_pixels` across: points read by the planes x and y, weighed by their
    volume."""
    mv = vertices.context.multivector
    return Camera(mv.x, mv.y, 0 * mv.x, mv.v, 0.0, np.array([period / 2, 0.5]),
                  np.array([period / 2 + margin, 0.5 + margin]), n_pixels)


def field(points: Extensor, faces: Topology, n_across: int, camera: Camera, values: np.ndarray,
          colour_map) -> np.ndarray:
    """A field at the tank's vertices `[vertices]`, a half above the middle's at the top and below at the bottom,
    coloured by its map, between the floor and the ceiling in the colours of what they hold, as bytes
    `[rows, columns, 3]`."""
    shown = closing(values, n_across)
    rows = points.reshape(-1, n_across)
    n_points = rows.shape[0]
    wall_colours = np.concatenate([np.repeat(np.array(colour_map(0.5 - span / 2))[None, :3], n_points - 1, axis=0),
                                   np.repeat(np.array(colour_map(0.5 + span / 2))[None, :3], n_points - 1, axis=0)])
    walls = simplices.strokes(concatenate([rows[:, 0], rows[:, -1]]), Topology.chains([n_points, n_points]),
                              wall_colours, wall_width, camera)
    colours = colour_map(np.clip(0.5 + span * shown, 0, 1))[:, :3]
    return on_white(simplices.composited([contours.picture(points, faces, shown, colours, camera), walls]))


def stacked(top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    """Two pictures of one width, the first above the second, with white rows between them."""
    return np.concatenate([top, np.full((n_gap,) + top.shape[1:], 255, dtype=np.uint8), bottom])


# --- drawing --------------------------------------------------------------------------
def draw_tank(topology: Topology, vertices: Extensor, n_across: int, period: float, temperature: Extensor,
              salinity: Extensor) -> Figure:
    """The tank's temperature above its salinity `Scalar+[vertices]`."""
    points, faces = unrolled(topology, vertices, n_across, period)
    camera = tank_camera(vertices, period)
    return image_figure(stacked(field(points, faces, n_across, camera, temperature.to_array(), temperature_map),
                                field(points, faces, n_across, camera, salinity.to_array(), salinity_map)))


def animate(topology: Topology, vertices: Extensor, n_across: int, period: float,
            frames: Iterable[tuple[Extensor, Extensor, Extensor]]) -> Iterator[np.ndarray]:
    """The tank at every frame of its stream function, temperature and salinity `Scalar+[vertices]` as it comes: its
    temperature above its salinity."""
    points, faces = unrolled(topology, vertices, n_across, period)
    camera = tank_camera(vertices, period)
    for _, temperature, salinity in frames:
        yield stacked(field(points, faces, n_across, camera, temperature.to_array(), temperature_map),
                      field(points, faces, n_across, camera, salinity.to_array(), salinity_map))


@style()
def draw_growth(ratios: np.ndarray, growth: Extensor, salt_diffusivity: float) -> Figure:
    """The growth rate of a small disturbance `[ratios] Scalar+` against the density ratio, with its zero, and the
    ratio of the diffusivities marked, beyond which no disturbance grows."""
    figure, ax = subplots(chart_width, chart_height)
    ax.axhline(0, color=axis_colour, linewidth=0.8)
    ax.axvline(1 / salt_diffusivity, color=axis_colour, linewidth=0.8, linestyle=":")
    ax.plot(ratios, growth.to_array(), "o-", color=growth_colour)
    ax.set_xticks([ratios[0], 1 / salt_diffusivity, ratios[-1]])
    ax.set_xlabel("density ratio")
    ax.set_ylabel("growth rate")
    figure.tight_layout(pad=0.4)
    return figure
