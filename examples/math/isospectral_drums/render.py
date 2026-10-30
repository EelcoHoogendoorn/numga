"""Triangular drum assemblies and flat maps of their signed displacement."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from functools import cache

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import LinearSegmentedColormap

from numga import stack
from examples.math.isospectral_drums.core import Point, Scalar, ga, mv
from examples.math.isospectral_drums.numerics import Mesh
from examples.util.wgsl import Shader

pixels = 260
# Displacement one way blue, the other orange, and rest a light neutral; the tiles' outlines.
colours = LinearSegmentedColormap.from_list(
    "membrane", ["#277daf", "#f1efe9", "#e79a47"],
)
outline = np.array([39, 49, 58]) / 255
# The width of the tiles' outlines in pixels.
outline_width = 1.4
tile_colours = np.array([
    [0.85, 0.40, 0.25], [0.94, 0.66, 0.27], [0.84, 0.78, 0.48],
    [0.35, 0.66, 0.54], [0.32, 0.65, 0.76], [0.39, 0.48, 0.72],
    [0.64, 0.45, 0.69],
])


# --- plumbing -------------------------------------------------------------------------
def coordinates(points: Point) -> np.ndarray:
    """Cartesian positions with field sites exposed for drawing: their pairings with the planes `mv.x` and `mv.y`."""
    return (stack((mv.x, mv.y)) & points.batch()[..., None]).to_array()


def amplitudes(field: Scalar) -> np.ndarray:
    """One signed displacement per vertex."""
    return np.asarray(field.batch().to_array())


def welded(corners: Point) -> tuple[Mesh, Point]:
    """The mesh of tiles `[tiles, corners]` whose separate tiles carry separate copies of a shared corner, each set of
    copies, at no distance from each other, welded into one vertex, and the welded vertices `[vertices]`."""
    copies = corners.reshape(-1)                                               # [copies] Point-
    together = (copies[:, None] - copies[None]).norm() < 1e-9                  # [copies, copies]
    # Each copy welded into the first copy at its place.
    welds, firsts, vertices = np.unique(together.argmax(axis=0), return_index=True, return_inverse=True)
    return Mesh(len(welds), vertices.reshape(corners.shape)), copies[firsts]


def draw_tiles(tiles: Point) -> plt.Figure:
    """The two polygonal drums, with matching colours for their seven constituent triangles."""
    points = coordinates(tiles)
    figure, axes = plt.subplots(1, len(points), figsize=(8, 3.7), facecolor="white")
    span = np.ptp(points, axis=(1, 2)).max() * 1.08
    for ax, drum, corners in zip(axes, tiles, points):
        ax.add_collection(PolyCollection(corners, facecolors=tile_colours,
                                        edgecolors="white", linewidths=1.5))
        mesh, welds = welded(drum)
        ax.add_collection(LineCollection(coordinates(welds[mesh.edges[mesh.boundary_edges]]), colors="#27313a",
                                         linewidths=1.5))
        centre = (corners.min(axis=(0, 1)) + corners.max(axis=(0, 1))) / 2
        ax.set(xlim=centre[0] + np.array([-0.5, 0.5]) * span,
               ylim=centre[1] + np.array([-0.5, 0.5]) * span, aspect="equal")
        ax.set_axis_off()
    figure.subplots_adjust(0, 0, 1, 1, wspace=0.07)
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
    @location(0) @interpolate(perspective, sample) value: f32,
    @location(1) @interpolate(perspective, sample) shares: vec3<f32>,
    @location(2) @interpolate(flat) rim: vec3<f32>,
}

@vertex
fn place(@builtin(vertex_index) index: u32) -> Corner {
    let face = faces[index / 3];
    let tile = tiles[owners[index / 3]];
    let point = points[face[index % 3]];
    // The point's share of each of its tile's corners: the area it spans with the other two, over the tile's. Each
    // is nothing on the edge opposite its corner and grows evenly away from it.
    let area = (tile.b - tile.a) & (tile.c - tile.a);
    let shares = vec3<f32>((tile.b - point) & (tile.c - point), (tile.c - point) & (tile.a - point),
                           (tile.a - point) & (tile.b - point)) / area;
    let offset = point - centre;
    return Corner(vec4<f32>((mv.x & offset) * scale, (mv.y & offset) * scale, 0.5, 1.0), values[face[index % 3]],
                  shares, tile.rim);
}

// The displacement through the colour table, inked where it crosses zero, a line wherever it lies within so many of
// its own changes across a pixel of zero, and outlined along the tiles' edges.
@fragment
fn shade(corner: Corner) -> @location(0) vec4<f32> {
    let entries = f32(textureDimensions(colours).x);
    let place = clamp(0.5 + corner.value / (2 * exposure), 0.0, 1.0) * (entries - 1) + 0.5;
    let colour = textureSampleLevel(colours, filtered, vec2<f32>(place / entries, 0.5), 0.0).rgb;
    let change = fwidth(corner.value);
    let nodal = clamp(1 - abs(corner.value) / (0.65 * change), 0.0, 1.0) * select(0.0, 1.0, change > 1e-12);
    // Each share over how fast it grows from pixel to pixel is how many pixels the place lies from the edge where it
    // is nothing; the outline fades out over the pixel about its reach.
    let growth = sqrt(dpdx(corner.shares) * dpdx(corner.shares) + dpdy(corner.shares) * dpdy(corner.shares));
    let edges = clamp(corner.rim + 0.5 - corner.shares / growth, vec3<f32>(0.0), vec3<f32>(1.0));
    return vec4<f32>(mix(colour * (1 - 0.4 * nodal), outline, max(edges.x, max(edges.y, edges.z))), 1.0);
}
"""


@cache
def membrane_shader() -> Shader:
    return Shader(membrane_source, ga, dict(Point=Point), dict(outline=outline), targets=("rgba32float",))


class Membrane:
    """A flat triangular mesh of tiles drawn on the GPU at a fixed place and scale, `pixels` across, with the tiles'
    outlines; the mesh's triangles come tile after tile, as many to each."""

    def __init__(self, vertices: Point, triangles: np.ndarray, corners: Point, scale: float):
        self.vertices, self.triangles, self.corners, self.scale = vertices.batch(), triangles, corners, scale
        axes = stack((mv.x, mv.y))
        readings = axes[:, None] & vertices.batch()
        low = readings.min(axis=-1)
        high = readings.max(axis=-1)
        # The middle of the drum's extent, a point as the vertices are: the plane's points, offsets from the origin,
        # as far along x and y as they read on the planes x and y.
        self.centre = (axes.dual() * ((low + high) / 2)).sum()
        # The outline is whole on the drum's perimeter, and half on each side of a seam between two tiles.
        n_tiles = corners.shape[0]
        tiling, _ = welded(corners)
        perimeter = np.isin(tiling.face_edges, tiling.boundary_edges)                      # [tiles, corners]
        self.rims = np.where(perimeter, outline_width, outline_width / 2)               # [tiles, corners]
        self.owners = np.repeat(np.arange(n_tiles), len(triangles) // n_tiles).astype(np.uint32)

    def draw(self, field: np.ndarray, exposure: float) -> np.ndarray:
        """Signed displacement with dark zero curves and the tiles' outlines, on white."""
        tiles = dict(a=self.corners[:, 0], b=self.corners[:, 1], c=self.corners[:, 2], rim=self.rims)
        drawn, = membrane_shader().draw((pixels, pixels), 3 * len(self.triangles), points=self.vertices,
                                       faces=self.triangles, values=field, tiles=tiles, owners=self.owners,
                                       centre=self.centre, scale=2 * self.scale / pixels, exposure=exposure,
                                       colours=colours(np.linspace(0, 1, 256))[None, :, :3])
        shown = drawn[..., :3] + (1 - drawn[..., 3:])
        return np.rint(255 * np.clip(shown, 0, 1)).astype(np.uint8)


def frames(
    vertices: tuple[Point, Point], meshes: tuple[Mesh, Mesh], tiles: Point,
    fields: Iterable[tuple[Scalar, Scalar]], exposure: float,
) -> Iterator[np.ndarray]:
    """Two flat displacement maps sharing a fixed scale and colour range, each with the outlines of its
    seven triangles `tiles [drums, tiles, corners]`."""
    scale = 0.91 * pixels / max(np.ptp(coordinates(value), axis=0).max() for value in vertices)
    membranes = tuple(Membrane(value, mesh.faces, drum, scale) for value, mesh, drum in zip(vertices, meshes, tiles))
    for pair in fields:
        yield np.concatenate([membrane.draw(amplitudes(field), exposure) for membrane, field in zip(membranes, pair)],
                             axis=1)


def draw_pieces(vertices: Point, mesh: Mesh, pieces: Scalar, exposure: float) -> plt.Figure:
    """Signed source contributions and their sum on the same reference triangle."""
    points = coordinates(vertices)
    values = amplitudes(pieces)
    edges = points[mesh.edges[mesh.boundary_edges]]
    centre = (points.min(axis=0) + points.max(axis=0)) / 2
    span = np.ptp(points, axis=0).max() * 1.05
    figure, axes = plt.subplots(1, len(values), figsize=(10, 2.5), facecolor="white")
    for ax, field in zip(axes, values):
        ax.tripcolor(points[:, 0], points[:, 1], mesh.faces, field, shading="gouraud",
                     cmap=colours, vmin=-exposure, vmax=exposure)
        contour_field = np.where(np.abs(field) < np.max(np.abs(field)) * 1e-10, 0, field)
        ax.tricontour(points[:, 0], points[:, 1], mesh.faces, contour_field, levels=[0],
                      colors="#394b59", linewidths=0.65, alpha=0.7)
        ax.add_collection(LineCollection(edges, colors="#27313a", linewidths=1.0))
        ax.set(xlim=centre[0] + np.array([-0.5, 0.5]) * span,
               ylim=centre[1] + np.array([-0.5, 0.5]) * span, aspect="equal")
        ax.set_axis_off()
    figure.subplots_adjust(0, 0, 1, 1, wspace=0.08)
    return figure
