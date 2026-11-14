"""Drawing for salt fingers: the tank's temperature and salinity at the start and over time.

A periodic tank closed along its length is drawn opened at its seam, repeating its first column one period further along. Its points are read by the coordinate planes `x` and `y` and weighed by geometry's projective plane `w & point`. Temperature is stacked above salinity, warm water in red above cold in blue and salty water in brown above fresh in green, framed by floor and ceiling strokes coloured by the values they hold.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np
from matplotlib import colormaps
from matplotlib.figure import Figure

from numga.extensor import Extensor
from numga.render import Camera, contours, drawing, simplices
from numga.sparse import Sparse
from numga.topology import Topology
from examples.fluids.saltfingers import core
from examples.util.panels import image_figure

mv = core.mv
temperature_map, salinity_map = colormaps["RdBu_r"], colormaps["BrBG_r"]
# Pixel width, margin in tank depths, gap rows between panels, wall stroke width, colour-map span, and odd contour band count.
n_pixels, margin, n_gap = 320, 0.03, 6
wall_width = 2.5
span = 0.8
n_bands = 9


# --- plumbing -------------------------------------------------------------------------
def white_composite(picture: np.ndarray) -> np.ndarray:
    """Premultiplied RGBA pixels composited over a white background as RGB bytes."""
    shown = picture[..., :3] + (1 - picture[..., 3:])
    return np.round(np.clip(shown, 0, 1) * 255).astype(np.uint8)


def tank_camera(period: float) -> Camera:
    """Side-on camera reading tank points by planes `x` and `y` at projective weight `w`."""
    return Camera(
        mv.x,
        mv.y,
        0 * mv.x,
        mv.w,
        0.0,
        np.array([period / 2, 0.5]),
        np.array([period / 2 + margin, 0.5 + margin]),
        n_pixels,
    )


def band_panel(topology: Topology, points: Extensor, camera: Camera, values: np.ndarray, colour_map) -> np.ndarray:
    """Banded contour image of a vertex scalar field between floor and ceiling wall strokes."""
    border = topology.border
    tails, heads = border.incidence[1, 0] * points                            # Point-[E_border], Point-[E_border]
    heights = (mv.y & tails).to_array()
    level = np.isclose(heights, (mv.y & heads).to_array())
    wall_colours = colour_map(0.5 + span * (heights[level] - 0.5))[:, :3]
    walls = simplices.strokes(points, border.subset(level, 1), wall_colours, wall_width, camera)
    # Evenly spaced flat colour bands with a white central band around zero.
    spacing = 1 / (n_bands - 1)
    middles = np.linspace(-0.5, 0.5, n_bands)
    bands = contours.Bands(-0.5 - spacing / 2, spacing, colour_map(0.5 + span * middles))
    return white_composite(
        drawing.composited([contours.picture(points, topology, values, np.ones(3), camera, bands=(bands,)), walls])
    )


def vertical_pair(top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    """Two equal-width RGB images separated by `n_gap` white pixel rows."""
    return np.concatenate([top, np.full((n_gap,) + top.shape[1:], 255, dtype=np.uint8), bottom])


# --- drawing --------------------------------------------------------------------------
def draw_tank(
    topology: Topology,
    vertices: Extensor,
    opening: Sparse,
    period: float,
    temperature: Extensor,
    salinity: Extensor,
) -> Figure:
    """Figure showing the opened tank's temperature panel above its salinity panel."""
    camera = tank_camera(period)
    return image_figure(
        vertical_pair(
            band_panel(topology, vertices, camera, (opening * temperature).to_array(), temperature_map),
            band_panel(topology, vertices, camera, (opening * salinity).to_array(), salinity_map),
        )
    )


def animate(
    topology: Topology,
    vertices: Extensor,
    opening: Sparse,
    period: float,
    frames: Iterable[tuple[Extensor, Extensor, Extensor]],
) -> Iterator[np.ndarray]:
    """Yield stacked temperature and salinity RGB frames for a stream of closed-tank states."""
    camera = tank_camera(period)
    for _, temperature, salinity in frames:
        yield vertical_pair(
            band_panel(topology, vertices, camera, (opening * temperature).to_array(), temperature_map),
            band_panel(topology, vertices, camera, (opening * salinity).to_array(), salinity_map),
        )
