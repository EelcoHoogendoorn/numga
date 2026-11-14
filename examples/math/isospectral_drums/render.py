"""GPU membrane shaders and planar panels render tiled drums, local triangle contributions, and vibrating displacement fields."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from functools import cache

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb

from numga import stack
from numga.topology import Topology
from numga.wgsl import Shader
from numga.render import contours, simplices
from numga.render.drawing import Draw, Target, drawn
from examples.util.panels import offset_panel, overlay
from examples.math.isospectral_drums.core import Point, Scalar, ga, mv

n_pixels = 260
colours = LinearSegmentedColormap.from_list(
    "membrane", ["#277daf", "#f1efe9", "#e79a47"],
)
outline = np.array([39, 49, 58]) / 255
outline_width = 1.4
rim_width, seam_width, piece_rim_width = 2.1, 2.1, 1.4
nodal_ink = np.append(to_rgb("#394b59"), 0.7)
nodal_width = 0.9
tile_colours = np.array([
    [0.85, 0.40, 0.25], [0.94, 0.66, 0.27], [0.84, 0.78, 0.48],
    [0.35, 0.66, 0.54], [0.32, 0.65, 0.76], [0.39, 0.48, 0.72],
    [0.64, 0.45, 0.69],
])


# --- plumbing -------------------------------------------------------------------------
def extent(points: Point) -> tuple[Scalar, Scalar]:
    """Coordinate planes bound `points` between minimum and maximum readings (`[2] Scalar+`, `[2] Scalar+`)."""
    readings = stack((mv.x, mv.y))[:, None] & points.batch().reshape(-1)        # [2, ...] Scalar+
    return readings.min(axis=-1), readings.max(axis=-1)


def amplitudes(field: Scalar) -> np.ndarray:
    """Signed scalar displacements unpack `field` into one float array (`[...]`) for plotting."""
    return np.asarray(field.batch().to_array())


def welded_tiling(corners: Point) -> tuple[Topology, Point]:
    """Coincident corners across `[..., 3]` triangle tiles merge into one shared topology and vertex array (`[...] Point-`)."""
    copies = corners.reshape(-1)                                                # [...] Point-
    matches = (copies[:, None] - copies[None]).antinorm() < 1e-9
    welds, firsts, vertex_indices = np.unique(matches.argmax(axis=0), return_index=True, return_inverse=True)
    return Topology(len(welds), vertex_indices.reshape(corners.shape)), copies[firsts]


def tiles_figure(tiles: Point) -> plt.Figure:
    """Polygonal outlines and seven matched colours compare the `[2, n_tiles, 3]` triangle tilings of the two drums."""
    n_tiles = tiles.shape[1]
    # Separate corner indices per tile allow distinct flat colours on adjacent triangles.
    pieces = Topology(3 * n_tiles, np.arange(3 * n_tiles).reshape(n_tiles, 3))
    figure, axes = plt.subplots(1, len(tiles), figsize=(8, 3.7), facecolor="white")
    figure.subplots_adjust(0, 0, 1, 1, wspace=0.07)
    extents = [extent(drum_tiles) for drum_tiles in tiles]
    half_span = stack([(high - low).max() for low, high in extents]).max() * 1.08 / 2  # [] Scalar+
    for ax, drum_tiles, (low, high) in zip(axes, tiles, extents):
        centre = (low + high) / 2                                               # [2] Scalar+
        limits = stack((centre - half_span, centre + half_span), axis=-1).to_array()
        ax.set(xlim=limits[0], ylim=limits[1], aspect="equal")
        ax.set_axis_off()
        camera = offset_panel(ax, mv.x, mv.y)
        topology, welds = welded_tiling(drum_tiles)
        overlay(ax, [
            contours.picture(drum_tiles.reshape(-1), pieces, np.zeros(3 * n_tiles), np.repeat(tile_colours, 3, axis=0), camera),
            simplices.strokes(drum_tiles.reshape(-1), pieces, np.ones(3), seam_width, camera),
            simplices.strokes(welds, topology.border, outline, rim_width, camera),
        ])
    return figure


membrane_source = """
var<storage, read> points: array<Point>;
var<storage, read> faces: array<vec3<u32>>;
var<storage, read> values: array<f32>;
// The tiles, and the tile each face lies in.
var<storage, read> tiles: array<Tile>;
var<storage, read> owners: array<u32>;
// The drum's middle, and how far across the image a unit length reaches.
var<uniform> centre: Point;
var<uniform> scale: f32;
var<uniform> exposure: f32;
var colours: texture_2d<f32>;
var filtered: sampler;

// A tile's corners, and how far into it, in pixels, its outline reaches from the edge opposite each.
struct Tile {
    a: Point,
    b: Point,
    c: Point,
    rim: vec3<f32>,
}

// Interpolated at every sample of a pixel, so the colour and the inked lines are supersampled with the edges.
struct Corner {
    @builtin(position) position: vec4<f32>,
    @interpolate(perspective, sample) value: f32,
    @interpolate(perspective, sample) shares: vec3<f32>,
    @interpolate(flat) rim: vec3<f32>,
}

@vertex
fn place(@builtin(vertex_index) index: u32) -> Corner {
    let face = faces[index / 3];
    let tile = tiles[owners[index / 3]];
    let point = points[face[index % 3]];
    // Area ratios across the tile vanish on opposite edges and grow linearly inward.
    let area = (tile.b - tile.a) & (tile.c - tile.a);
    let shares = vec3<f32>(((tile.b - point) & (tile.c - point)) / area, ((tile.c - point) & (tile.a - point)) / area,
                           ((tile.a - point) & (tile.b - point)) / area);
    let offset = point - centre;
    return Corner(vec4<f32>((x_plane & offset) * scale, (y_plane & offset) * scale, 0.5, 1.0), values[face[index % 3]],
                  shares, tile.rim);
}

@fragment
fn shade(corner: Corner) -> @location(0) vec4<f32> {
    let entries = f32(textureDimensions(colours).x);
    let place = clamp(0.5 + corner.value / (2 * exposure), 0.0, 1.0) * (entries - 1) + 0.5;
    let colour = textureSampleLevel(colours, filtered, vec2<f32>(place / entries, 0.5), 0.0).rgb;
    let change = fwidth(corner.value);
    let nodal = clamp(1 - abs(corner.value) / (0.65 * change), 0.0, 1.0) * select(0.0, 1.0, change > 1e-12);
    // Screen-space derivatives convert area ratios into pixel distance from each tile edge.
    let growth = sqrt(dpdx(corner.shares) * dpdx(corner.shares) + dpdy(corner.shares) * dpdy(corner.shares));
    let edges = clamp(corner.rim + 0.5 - corner.shares / growth, vec3<f32>(0.0), vec3<f32>(1.0));
    return vec4<f32>(mix(colour * (1 - 0.4 * nodal), outline, max(edges.x, max(edges.y, edges.z))), 1.0);
}
"""


@cache
def membrane_shader() -> Shader:
    """Compiles the WGSL shader that colours membrane displacements and supersamples nodal lines and tile seams."""
    return Shader(membrane_source, ga, dict(Point=Point), dict(outline=outline, x_plane=mv.x, y_plane=mv.y))


class Membrane:
    """GPU renderer draws a drum triangulation of `n_pixels` by `n_pixels` pixels with antialiased nodal curves and tile outlines."""

    def __init__(self, vertices: Point, triangles: np.ndarray, corners: Point, scale: float):
        self.vertices, self.triangles, self.corners, self.scale = vertices.batch(), triangles, corners, scale
        low, high = extent(vertices)                                            # [2] Scalar+, [2] Scalar+
        # Midpoint of the drum's extent as an intrinsic planar point.
        self.centre = (stack((mv.x, mv.y)).dual() * ((low + high) / 2)).sum()   # [] Point-
        # Perimeter edges carry the full outline width; interior seams split it across adjacent tiles.
        n_tiles = corners.shape[0]
        tiling = welded_tiling(corners)[0]
        perimeter = np.isin(tiling.boundary_indices[1], tiling.exterior[1])
        self.rims = np.where(perimeter, outline_width, outline_width / 2)
        self.owners = np.repeat(np.arange(n_tiles), len(triangles) // n_tiles).astype(np.uint32)

    def image(self, field: np.ndarray, exposure: float) -> np.ndarray:
        """Renders one signed displacement field onto a white `[n_pixels, n_pixels, 3]` RGB canvas."""
        tiles = dict(a=self.corners[:, 0], b=self.corners[:, 1], c=self.corners[:, 2], rim=self.rims)
        membrane, = drawn([Draw(membrane_shader(), (n_pixels, n_pixels), 3 * len(self.triangles), dict(
            points=self.vertices, faces=self.triangles, values=field, tiles=tiles, owners=self.owners,
            centre=self.centre, scale=2 * self.scale / n_pixels, exposure=exposure,
            colours=colours(np.linspace(0, 1, 256))[None, :, :3]))], Target(("rgba32float",), False), 1)
        shown = membrane[..., :3] + (1 - membrane[..., 3:])
        return np.rint(255 * np.clip(shown, 0, 1)).astype(np.uint8)


def drum_frames(
    vertices: tuple[Point, Point], topologies: tuple[Topology, Topology], tiles: Point,
    exposure: float, states: Iterable[tuple[Scalar, Scalar]],
) -> Iterator[np.ndarray]:
    """Yields side-by-side `[n_pixels, 2 * n_pixels, 3]` RGB displacement maps for the two tiled drums at shared scale and colour range."""
    widest = stack([(high - low).max() for low, high in map(extent, vertices)]).max()  # [] Scalar+
    scale = float((0.91 * n_pixels / widest).to_array())
    membranes = tuple(
        Membrane(drum_vertices, topology.elements[2], drum_tiles, scale)
        for drum_vertices, topology, drum_tiles in zip(vertices, topologies, tiles)
    )
    for pair in states:
        yield np.concatenate([membrane.image(amplitudes(field), exposure) for membrane, field in zip(membranes, pair)], axis=1)


def pieces_figure(vertices: Point, topology: Topology, pieces: Scalar, exposure: float) -> plt.Figure:
    """Four panels compare the three signed source triangle contributions and their sum (`[4] Scalar+[V]`) on the reference triangle."""
    values = amplitudes(pieces)
    low, high = extent(vertices)                                                # [2] Scalar+, [2] Scalar+
    centre, half_span = (low + high) / 2, (high - low).max() * 1.05 / 2         # [2] Scalar+, [] Scalar+
    limits = stack((centre - half_span, centre + half_span), axis=-1).to_array()
    places = vertices.batch()                                                   # [...] Point-
    rim = topology.border
    figure, axes = plt.subplots(1, len(values), figsize=(10, 2.5), facecolor="white")
    figure.subplots_adjust(0, 0, 1, 1, wspace=0.08)
    for ax, field in zip(axes, values):
        ax.set(xlim=limits[0], ylim=limits[1], aspect="equal")
        ax.set_axis_off()
        camera = offset_panel(ax, mv.x, mv.y)
        shades = colours(Normalize(-exposure, exposure)(field))[:, :3]
        nodal_lines = contours.Level(0.0, nodal_ink, nodal_width)
        overlay(ax, [
            contours.picture(places, topology, field, shades, camera, levels=(nodal_lines,)),
            simplices.strokes(places, rim, outline, piece_rim_width, camera),
        ])
    return figure
