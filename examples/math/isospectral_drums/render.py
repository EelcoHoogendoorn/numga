"""Triangular drum assemblies and flat maps of their signed displacement."""

from __future__ import annotations

from collections.abc import Iterable
from functools import cache

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image, ImageDraw

from numga import stack
from examples.math.isospectral_drums.core import Point, Scalar, ga, mv
from examples.wgsl import Shader

PIXELS = 380
# Displacement one way blue, the other orange, and rest a light neutral; the tiles' outlines.
COLOURS = LinearSegmentedColormap.from_list(
    "membrane", ["#277daf", "#f1efe9", "#e79a47"],
)
OUTLINE = (39, 49, 58)
TILE_COLOURS = np.array([
    [0.85, 0.40, 0.25], [0.94, 0.66, 0.27], [0.84, 0.78, 0.48],
    [0.35, 0.66, 0.54], [0.32, 0.65, 0.76], [0.39, 0.48, 0.72],
    [0.64, 0.45, 0.69],
])


# --- plumbing -------------------------------------------------------------------------
def coordinates(points: Point) -> np.ndarray:
    """Cartesian positions with field sites exposed for drawing: their pairings with the lines x and y."""
    return (stack((mv.x, mv.y)) & points.batch()[..., None]).to_array()


def amplitudes(field: Scalar) -> np.ndarray:
    """One signed displacement per vertex."""
    return np.asarray(field.batch().to_array())


def boundary_corners(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """The edge opposite each triangle corner, marked when it belongs to the perimeter."""
    # Separate tiles may carry separate copies of a shared edge's vertices.
    _, vertices = np.unique(np.round(points, 12), axis=0, return_inverse=True)
    edges = np.sort(vertices[triangles[:, [[1, 2], [2, 0], [0, 1]]]], axis=-1)
    _, inverse, counts = np.unique(edges.reshape(-1, 2), axis=0, return_inverse=True, return_counts=True)
    return (counts[inverse] == 1).reshape(-1, 3)


def draw_tiles(tiles: Point) -> plt.Figure:
    """The two polygonal drums, with matching colours for their seven constituent triangles."""
    points = coordinates(tiles)
    figure, axes = plt.subplots(1, len(points), figsize=(8, 3.7), facecolor="white")
    span = np.ptp(points, axis=(1, 2)).max() * 1.08
    for ax, corners in zip(axes, points):
        ax.add_collection(PolyCollection(corners, facecolors=TILE_COLOURS,
                                        edgecolors="white", linewidths=1.5))
        faces = np.arange(corners.shape[0] * 3).reshape(-1, 3)
        boundary = boundary_corners(corners.reshape(-1, 2), faces)
        edges = corners[:, [[1, 2], [2, 0], [0, 1]]][boundary]
        ax.add_collection(LineCollection(edges, colors="#27313a", linewidths=1.5))
        centre = (corners.min(axis=(0, 1)) + corners.max(axis=(0, 1))) / 2
        ax.set(xlim=centre[0] + np.array([-0.5, 0.5]) * span,
               ylim=centre[1] + np.array([-0.5, 0.5]) * span, aspect="equal")
        ax.set_axis_off()
    figure.subplots_adjust(0, 0, 1, 1, wspace=0.07)
    plt.close(figure)
    return figure


membrane_source = """
var<storage, read> points: array<Point>;
var<storage, read> faces: array<vec3<u32>>;
var<storage, read> values: array<f32>;
// Per face: one where the edge opposite each corner lies on the drum's perimeter.
var<storage, read> rims: array<vec3<f32>>;
// The drum's middle, and how far across the image a unit length reaches.
var<uniform> centre: Point;
var<uniform> scale: f32;
var<uniform> exposure: f32;
var colours: texture_2d<f32>;
var filtered: sampler;

// Interpolated at every sample of a pixel, so the colour and the inked lines are supersampled with the edges.
struct Corner {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(perspective, sample) value: f32,
    @location(1) @interpolate(perspective, sample) corner: vec3<f32>,
    @location(2) @interpolate(flat) rim: vec3<f32>,
}

@vertex
fn place(@builtin(vertex_index) index: u32) -> Corner {
    let face = faces[index / 3];
    let which = index % 3;
    let offset = points[face[which]] - centre;
    let corner = vec3<f32>(f32(which == 0), f32(which == 1), f32(which == 2));
    return Corner(vec4<f32>((mv.x & offset) * scale, (mv.y & offset) * scale, 0.5, 1.0), values[face[which]], corner, rims[index / 3]);
}

// The displacement through the colour table, inked where it crosses zero and along the perimeter: a line a
// pixel or so wide wherever a quantity lies within that many of its own changes across a pixel of zero.
@fragment
fn shade(corner: Corner) -> @location(0) vec4<f32> {
    let entries = f32(textureDimensions(colours).x);
    let place = clamp(0.5 + corner.value / (2 * exposure), 0.0, 1.0) * (entries - 1) + 0.5;
    let colour = textureSampleLevel(colours, filtered, vec2<f32>(place / entries, 0.5), 0.0).rgb;
    let change = fwidth(corner.value);
    let nodal = clamp(1 - abs(corner.value) / (0.65 * change), 0.0, 1.0) * select(0.0, 1.0, change > 1e-12);
    let edges = clamp(1 - corner.corner / (1.3 * fwidth(corner.corner)), vec3<f32>(0.0), vec3<f32>(1.0)) * corner.rim;
    let ink = max(0.4 * nodal, 0.78 * max(edges.x, max(edges.y, edges.z)));
    return vec4<f32>(colour * (1 - ink), 1.0);
}
"""


@cache
def membrane_shader() -> Shader:
    return Shader(membrane_source, ga, dict(Point=Point), targets=("rgba32float",))


class Membrane:
    """A flat triangular mesh drawn on the GPU at a fixed place and scale, `pixels` across."""

    def __init__(self, vertices: Point, points: np.ndarray, triangles: np.ndarray, scale: float):
        self.vertices, self.triangles = vertices.batch(), triangles
        self.centre_point = (points.min(axis=0) + points.max(axis=0)) / 2
        self.centre = mv.x.dual() * self.centre_point[0] + mv.y.dual() * self.centre_point[1]
        self.scale = scale
        self.rims = boundary_corners(points, triangles).astype(float)

    def draw(self, field: np.ndarray, exposure: float) -> np.ndarray:
        """Signed displacement with dark zero curves, on white."""
        drawn, = membrane_shader().draw((PIXELS, PIXELS), 3 * len(self.triangles), points=self.vertices,
                                       faces=self.triangles, values=field, rims=self.rims, centre=self.centre,
                                       scale=2 * self.scale / PIXELS, exposure=exposure,
                                       colours=COLOURS(np.linspace(0, 1, 256))[None, :, :3])
        shown = drawn[..., :3] + (1 - drawn[..., 3:])
        return np.rint(255 * np.clip(shown, 0, 1)).astype(np.uint8)

    def outline(self, picture: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """The picture with the outlines of triangles `[tiles, 3, 2]` drawn over it."""
        screen = (corners - self.centre_point) * self.scale + PIXELS / 2
        screen[..., 1] = PIXELS - screen[..., 1]
        image = Image.fromarray(picture)
        draw = ImageDraw.Draw(image)
        for triangle in screen:
            draw.polygon([tuple(point) for point in triangle], outline=OUTLINE, width=2)
        return np.asarray(image)


def frames(
    vertices: tuple[Point, Point], triangles: tuple[np.ndarray, np.ndarray], tiles: Point,
    fields: Iterable[tuple[Scalar, Scalar]], exposure: float,
) -> list[np.ndarray]:
    """Two flat displacement maps sharing a fixed scale and colour range, each with the outlines of its
    seven triangles `tiles [2, tiles, 3]`."""
    points = tuple(coordinates(value) for value in vertices)
    corners = coordinates(tiles)                                             # [2, tiles, 3, 2]
    scale = 0.91 * PIXELS / max(np.ptp(value, axis=0).max() for value in points)
    membranes = tuple(Membrane(value, places, faces, scale) for value, places, faces in zip(vertices, points, triangles))
    return [np.concatenate([membrane.outline(membrane.draw(amplitudes(field), exposure), drum)
                            for membrane, field, drum in zip(membranes, pair, corners)], axis=1)
            for pair in fields]


def draw_pieces(vertices: Point, triangles: np.ndarray, pieces: Scalar, exposure: float) -> plt.Figure:
    """Signed source contributions and their sum on the same reference triangle."""
    points = coordinates(vertices)
    values = amplitudes(pieces)
    perimeter = boundary_corners(points, triangles)
    edges = points[triangles[:, [[1, 2], [2, 0], [0, 1]]]][perimeter]
    centre = (points.min(axis=0) + points.max(axis=0)) / 2
    span = np.ptp(points, axis=0).max() * 1.05
    figure, axes = plt.subplots(1, len(values), figsize=(10, 2.5), facecolor="white")
    for ax, field in zip(axes, values):
        ax.tripcolor(points[:, 0], points[:, 1], triangles, field, shading="gouraud",
                     cmap=COLOURS, vmin=-exposure, vmax=exposure)
        contour_field = np.where(np.abs(field) < np.max(np.abs(field)) * 1e-10, 0, field)
        ax.tricontour(points[:, 0], points[:, 1], triangles, contour_field, levels=[0],
                      colors="#394b59", linewidths=0.65, alpha=0.7)
        ax.add_collection(LineCollection(edges, colors="#27313a", linewidths=1.0))
        ax.set(xlim=centre[0] + np.array([-0.5, 0.5]) * span,
               ylim=centre[1] + np.array([-0.5, 0.5]) * span, aspect="equal")
        ax.set_axis_off()
    figure.subplots_adjust(0, 0, 1, 1, wspace=0.08)
    plt.close(figure)
    return figure
