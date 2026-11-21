"""Rendering for snowflake growth draws pale icy blue where a parcel's ice fraction passes one half, with a crisp white
edge, over liquid water tinted by its temperature from deep blue in the cold water to lighter blue where latent heat has
warmed it, and holds the final crystal briefly before the loop repeats. A close-up shows the seed inside the
triangulated disc, and a chart plots the total entropy gained over time.

The dish's vertex positions are unit-weight places read by pairing with `x` and `y` against weight `w` in `Camera`,
while a parcel's ice and heat are read by pairing with `i` and `h`.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb, to_rgba
from matplotlib.figure import Figure

from numga import dim, stack
from numga.extensor import Extensor
from examples.util import charts
from numga.topology import Topology
from numga.render import Camera, contours, drawing, simplices
from examples.util.panels import image_figure
from examples.thermodynamics.snowflake import core

mv = core.mv
# Water colour palette from cold water up to the melting temperature, plus ice fill and contour colours.
water_map = LinearSegmentedColormap.from_list("water", ["#0b1a2e", "#1d3a5f", "#4f7cab"])
ice_colour, edge_colour = np.array(to_rgba("#dcefff")), np.array(to_rgba("#ffffff"))
# Melting temperature at which the water palette reaches its lightest tint.
warmest = mv.scalar(1.0, unit=dim.temperature)
# Image width in pixels, viewport padding factor over the dish radius, contour width, and side-by-side gap in pixels.
n_pixels, view_share, edge_width, gap = 300, 1.04, 1.0, 8
# Number of trailing frames holding the grown crystal before the loop restarts.
n_hold = 15
# Seed close-up half-width and triangle stroke styling.
seed_half = mv.scalar(0.3, unit=dim.length)
stroke_colour, stroke_width, stroke_opacity = np.array(to_rgb("#8fa8c8")), mv.scalar(1.0, unit=dim.pixel), 0.6
# Entropy chart dimensions and colours.
entropy_size, entropy_colour, axis_colour = (300, 180), "#1d3a5f", "#555555"


def on_white(picture: np.ndarray) -> np.ndarray:
    """RGBA colour and coverage composited over white as uint8 RGB."""
    shown = picture[..., :3] + (1 - picture[..., 3:])
    return np.round(np.clip(shown, 0, 1) * 255).astype(np.uint8)


def layers(topology: Topology, points: Extensor, parcels: Extensor, camera: Camera) -> list:
    """Compositing layers for the water temperature tint and the ice contour and fill across the dish."""
    # Fill pale blue above ice fraction 0.5 and draw a crisp contour line along 0.5.
    ice_band = contours.Bands(-0.5, 1.0, np.array([np.zeros(4), ice_colour]))
    edge = contours.Level(0.5, edge_colour, edge_width)
    tint = water_map(np.clip(np.asarray(((mv.h & parcels) / warmest).to_array()), 0, 1))[:, :3]
    fractions = np.asarray((mv.i & parcels).to_array())
    return [contours.picture(points, topology, fractions, tint, camera),
            contours.picture(points, topology, fractions, tint, camera, levels=(edge,), bands=(ice_band,), opacity=0.0)]


def framed(half: Extensor) -> Camera:
    """Orthographic camera reading `x` and `y` on unit-weight places around the origin into pixels."""
    half_pixels = mv.scalar(n_pixels / 2, unit=dim.pixel)
    scale = half_pixels / half
    middle = mv.scalar([[0.0], [0.0]], unit=dim.pixel)
    return Camera(scale * mv.x, scale * mv.y, mv.zeros(mv.x.gatype), mv.w, 0.0,
                  middle, stack((half_pixels, half_pixels)))


def dish_camera(vertices: Extensor) -> Camera:
    """Camera framing the full dish with a thin white margin around the rim."""
    points = vertices.batch()
    return framed(view_share * (mv.x & points).abs().max())


def animate(topology: Topology, vertices: Extensor,
            states: Iterable[tuple[Extensor, Extensor]]) -> Iterator[np.ndarray]:
    """Yield RGB frames for each recorded parcel and phase state, holding the final frame briefly at the end."""
    points = vertices.batch()
    camera = dish_camera(vertices)
    for parcels, _ in states:
        shown = on_white(drawing.composited(layers(topology, points, parcels, camera)))
        yield shown
    for _ in range(n_hold):
        yield shown


def draw_grown(topology: Topology, vertices: Extensor, parcels: Extensor) -> Figure:
    """Still figure of the grown crystal across the full dish."""
    camera = dish_camera(vertices)
    return image_figure(on_white(drawing.composited(layers(topology, vertices.batch(), parcels, camera))))


def draw_seed(topology: Topology, vertices: Extensor, parcels: Extensor) -> Figure:
    """Close-up figure of the initial ice seed and surrounding triangle edges."""
    points = vertices.batch()
    camera = framed(seed_half)
    strokes = simplices.strokes(points, topology, stroke_colour, stroke_width, camera, opacity=stroke_opacity)
    return image_figure(on_white(drawing.composited(layers(topology, points, parcels, camera) + [strokes])))


def side_by_side(left: list[np.ndarray], right: list[np.ndarray]) -> Iterator[np.ndarray]:
    """Yield side-by-side frame pairs separated by a white gap, holding the shorter sequence on its final frame."""
    spacer = np.full((left[0].shape[0], gap, 3), 255, dtype=np.uint8)
    for index in range(max(len(left), len(right))):
        yield np.concatenate([left[min(index, len(left) - 1)], spacer, right[min(index, len(right) - 1)]], axis=1)


@charts.style()
def draw_entropy(times: Extensor | np.ndarray, entropies: Extensor) -> Figure:
    """Chart of the total entropy gained since the start against time."""
    x_values = np.asarray(times.to_array()) if isinstance(times, Extensor) else times
    figure, axis = charts.subplots(*entropy_size, layout="constrained")
    axis.plot(x_values, (entropies - entropies[0]).to_array(), color=entropy_colour)
    axis.axhline(0.0, color=axis_colour, linewidth=charts.FRAME_WIDTH)
    axis.set(xlim=(x_values[0], x_values[-1]), xticks=[], yticks=[], xlabel="time", ylabel="entropy gained")
    axis.spines[["top", "right", "bottom"]].set_visible(False)
    axis.spines["left"].set_color(axis_colour)
    return figure
