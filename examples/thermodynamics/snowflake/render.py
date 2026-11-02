"""Drawing for a snowflake growing: the ice, where its fraction passes a half, in a pale icy blue with a crisp edge,
over the water tinted by its temperature, from the cold water's deep blue to a lighter blue where the latent heat
has warmed it; the finished crystal held a moment before the loop starts over. The seed is also drawn close up in the
mesh's triangles.

The dish's points are read by the planes x and y and weighed by `mv.v & point`, the algebra's own projective weight; a
parcel's ice and heat are read by `mv.i` and `mv.h`.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb, to_rgba
from matplotlib.figure import Figure

from numga.extensor import Extensor
from examples.util.wgsl import contours, image_figure, simplices
from examples.util.wgsl.camera import Camera

Topology = simplices.Topology
# The water's tint from cold to the melting temperature, the ice's fill and its edge.
water_map = LinearSegmentedColormap.from_list("water", ["#0b1a2e", "#1d3a5f", "#4f7cab"])
ice_colour, edge_colour = np.array(to_rgba("#dcefff")), np.array(to_rgba("#ffffff"))
# The temperature at which the water's tint is lightest, the melting temperature's.
warmest = 1.0
# The picture's pixels across; the share of the dish's radius it spans either way from the middle, inside the dish;
# the edge's width, and the white gap between pictures set side by side, in pixels.
n_pixels, view_share, edge_width, gap = 300, 0.7, 1.0, 8
# The frames the finished crystal is held for before the loop starts over.
n_hold = 15
# The close-up of the seed: its half width in the dish's lengths, and the mesh's strokes, their colour, width in
# pixels and opacity.
seed_half, mesh_colour, mesh_width, mesh_opacity = 0.3, np.array(to_rgb("#8fa8c8")), 1.0, 0.6


def on_white(picture: np.ndarray) -> np.ndarray:
    """Colour and coverage `[rows, columns, 4]` laid over white, as bytes `[rows, columns, 3]`."""
    shown = picture[..., :3] + (1 - picture[..., 3:])
    return np.round(np.clip(shown, 0, 1) * 255).astype(np.uint8)


def layers(topology: Topology, points: Extensor, parcels: Extensor, camera: Camera) -> list:
    """The ice of these parcels `Point-[vertices]` over the water's tint, at the dish's points `[vertices] Point-`: the
    layers of the picture."""
    mv = points.context.multivector
    # The ice where its fraction passes a half, the water below clear, and the edge where it does.
    ice_band = contours.Bands(-0.5, 1.0, np.array([np.zeros(4), ice_colour]))
    edge = contours.Level(0.5, edge_colour, edge_width)
    tint = water_map(np.clip(np.asarray((mv.h & parcels).to_array()) / warmest, 0, 1))[:, :3]
    fractions = np.asarray((mv.i & parcels).to_array())
    return [contours.picture(points, topology, fractions, tint, camera),
            contours.picture(points, topology, fractions, tint, camera, levels=(edge,), bands=(ice_band,), opacity=0.0)]


def framed(points: Extensor, half: float) -> Camera:
    """The camera looking down on the dish's points `[vertices] Point-` by the planes x and y, weighed by `mv.v`, this
    half width either way from the middle."""
    mv = points.context.multivector
    return Camera(mv.x, mv.y, 0 * mv.x, mv.v, 0.0, np.zeros(2), np.array([half, half]), n_pixels)


def animate(topology: Topology, vertices: Extensor, frames: Iterable[Extensor]) -> Iterator[np.ndarray]:
    """The dish at its vertices `Point-[vertices]` at every frame of its parcels `Point-[vertices]` as they come, their
    ice and their heat, the temperature: the ice over the water's tint; the last frame held."""
    mv = vertices.context.multivector
    points = vertices.batch()
    camera = framed(points, view_share * np.abs((mv.x & points).to_array()).max())
    for parcels in frames:
        shown = on_white(simplices.composited(layers(topology, points, parcels, camera)))
        yield shown
    for _ in range(n_hold):
        yield shown


def draw_seed(topology: Topology, vertices: Extensor, parcels: Extensor) -> Figure:
    """The seed of ice in these parcels `Point-[vertices]` close up, in the dish's triangles at its vertices
    `Point-[vertices]`."""
    points = vertices.batch()
    camera = framed(points, seed_half)
    strokes = simplices.strokes(points, topology, mesh_colour, mesh_width, camera, opacity=mesh_opacity)
    return image_figure(on_white(simplices.composited(layers(topology, points, parcels, camera) + [strokes])))


def side_by_side(pairs: Iterable[tuple[np.ndarray, np.ndarray]]) -> Iterator[np.ndarray]:
    """Two pictures of each frame `[rows, columns, 3]` side by side, a white gap between them."""
    for left, right in pairs:
        yield np.concatenate([left, np.full((left.shape[0], gap, 3), 255, dtype=np.uint8), right], axis=1)
