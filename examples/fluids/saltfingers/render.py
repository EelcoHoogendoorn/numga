"""Drawing for salt fingers: the tank's temperature and salinity, at the start and over time.

A tank closed on itself along its length is drawn opened at its seam, its first column again a period further along.
Its points are read by the planes x and y and weighed by
their volume, `mv.v & point`, the algebra's own projective weight. The temperature is drawn above the salinity, the same
tank in step: the temperature red above the middle's and blue below, the salinity brown above the middle's and green
below, with the walls strokes in the colours of what they hold.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np
from matplotlib import colormaps
from matplotlib.figure import Figure

from numga.extensor import Extensor
from examples.util.wgsl import contours, image_figure, simplices
from examples.util.wgsl.camera import Camera

Topology = simplices.Topology
temperature_map, salinity_map = colormaps["RdBu_r"], colormaps["BrBG_r"]
# The tank's picture: pixels across, the room around it in depths, the rows between the two fields, and the walls'
# width; the share of each colour map the fields span, short of its darkest ends, and the bands each field is drawn in,
# centred on evenly spaced levels from the bottom's to the top's, an odd count so that the middle is one band, white.
n_pixels, margin, n_gap = 320, 0.03, 6
wall_width = 2.5
span = 0.8
n_bands = 9


# --- plumbing -------------------------------------------------------------------------
def on_white(picture: np.ndarray) -> np.ndarray:
    """Colour and coverage `[rows, columns, 4]` laid over white, as bytes `[rows, columns, 3]`."""
    shown = picture[..., :3] + (1 - picture[..., 3:])
    return np.round(np.clip(shown, 0, 1) * 255).astype(np.uint8)


def tank_camera(vertices: Extensor, period: float) -> Camera:
    """The camera looking at the tank side on, `n_pixels` across: points read by the planes x and y, weighed by their
    volume."""
    mv = vertices.context.multivector
    return Camera(mv.x, mv.y, 0 * mv.x, mv.v, 0.0, np.array([period / 2, 0.5]),
                  np.array([period / 2 + margin, 0.5 + margin]), n_pixels)


def field(topology: Topology, points: Extensor, camera: Camera, values: np.ndarray, colour_map) -> np.ndarray:
    """A field at the open tank's vertices `[vertices]`, a half above the middle's at the top and below at the bottom,
    coloured by its map in bands, between the floor and the ceiling, the edges of the border that run along the tank,
    in the colours of what they hold, as bytes `[rows, columns, 3]`."""
    border = topology.border
    tails, heads = border.incidence[1, 0] * points                              # Point-[border], Point-[border]
    heights = (points.context.multivector.y & tails).to_array()                 # [border]
    level = np.isclose(heights, (points.context.multivector.y & heads).to_array())   # [border]
    wall_colours = colour_map(0.5 + span * (heights[level] - 0.5))[:, :3]       # [walls, 3]
    walls = simplices.strokes(points, border.subset(level, 1), wall_colours, wall_width, camera)
    # Each band in the colour of the field at its middle, flat across it.
    spacing = 1 / (n_bands - 1)
    middles = np.linspace(-0.5, 0.5, n_bands)
    bands = contours.Bands(-0.5 - spacing / 2, spacing, colour_map(0.5 + span * middles))
    return on_white(simplices.composited([contours.picture(points, topology, values, np.ones(3), camera, bands=(bands,)),
                                          walls]))


def stacked(top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    """Two pictures of one width, the first above the second, with white rows between them."""
    return np.concatenate([top, np.full((n_gap,) + top.shape[1:], 255, dtype=np.uint8), bottom])


# --- drawing --------------------------------------------------------------------------
def draw_tank(topology: Topology, vertices: Extensor, opening: Extensor, period: float, temperature: Extensor,
              salinity: Extensor) -> Figure:
    """The open tank's temperature above its salinity, the closed tank's `Scalar+[vertices]` read on it by
    `opening`."""
    camera = tank_camera(vertices, period)
    return image_figure(stacked(field(topology, vertices, camera, (opening * temperature).to_array(), temperature_map),
                                field(topology, vertices, camera, (opening * salinity).to_array(), salinity_map)))


def animate(topology: Topology, vertices: Extensor, opening: Extensor, period: float,
            frames: Iterable[tuple[Extensor, Extensor, Extensor]]) -> Iterator[np.ndarray]:
    """The open tank at every frame of the closed tank's stream function, temperature and salinity `Scalar+[vertices]`
    as it comes, read on the open tank by `opening`: its temperature above its salinity."""
    camera = tank_camera(vertices, period)
    for _, temperature, salinity in frames:
        yield stacked(field(topology, vertices, camera, (opening * temperature).to_array(), temperature_map),
                      field(topology, vertices, camera, (opening * salinity).to_array(), salinity_map))
