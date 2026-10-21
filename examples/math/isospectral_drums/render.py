"""Triangular drum assemblies and flat maps of their signed displacement."""

from __future__ import annotations

from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image, ImageDraw

from examples.math.isospectral_drums.core import Scalar, Vector, ga
from examples.raster import rasterise

PIXELS = 380
SUPERSAMPLE = 2
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
def coordinates(points: Vector) -> np.ndarray:
    """Cartesian positions with field sites exposed for drawing."""
    return np.asarray(points.batch().cast(ga.subspace("x y")).kernel)


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


def image_figure(picture: np.ndarray) -> plt.Figure:
    """An image at its pixel size without margins or axes."""
    figure, ax = plt.subplots(figsize=(picture.shape[1] / 100, picture.shape[0] / 100), dpi=100)
    ax.imshow(picture)
    ax.set_axis_off()
    figure.subplots_adjust(0, 0, 1, 1)
    plt.close(figure)
    return figure


def draw_tiles(tiles: Vector) -> plt.Figure:
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


class MembraneRaster:
    """Fixed pixel interpolation and boundary outlines for a flat triangular mesh."""

    def __init__(self, points: np.ndarray, triangles: np.ndarray, scale: float):
        size = PIXELS * SUPERSAMPLE
        centre = (points.min(axis=0) + points.max(axis=0)) / 2
        self.centre, self.scale = centre, scale
        screen = (points - centre) * scale + size / 2
        self.pixel, self.face, self.weights, _ = rasterise(
            np.pad(screen, ((0, 0), (0, 1))), triangles, size)
        self.triangles = triangles

        corners = screen[triangles]
        across = corners[:, 1:] - corners[:, :1]
        area = across[:, 0, 0] * across[:, 1, 1] - across[:, 0, 1] * across[:, 1, 0]
        opposite = np.roll(corners, -1, axis=1) - np.roll(corners, 1, axis=1)
        self.gradients = np.stack([opposite[..., 1], -opposite[..., 0]], axis=-1) / area[:, None, None]
        altitudes = np.abs(area[:, None]) / np.linalg.norm(opposite, axis=-1)
        edge_distance = self.weights * altitudes[self.face]
        perimeter = boundary_corners(points, triangles)
        self.rim = np.max(np.clip(1 - edge_distance / (1.3 * SUPERSAMPLE), 0, 1)
                          * perimeter[self.face], axis=-1)

    def draw(self, field: np.ndarray, exposure: float) -> np.ndarray:
        """Signed displacement with dark zero curves and a visible grey resting membrane."""
        corner_values = field[self.triangles]
        values = np.einsum("pk,pk->p", self.weights, corner_values[self.face])
        colours = COLOURS(np.clip(0.5 + values / (2 * exposure), 0, 1))[:, :3]

        # The scalar's slope gives a contour width measured in pixels, independent of amplitude.
        gradient = np.einsum("fkd,fk->fd", self.gradients, corner_values)
        slope = np.linalg.norm(gradient, axis=-1)[self.face]
        distance = np.abs(values) / np.maximum(slope, 1e-14)
        nodal_ink = np.clip(1 - distance / (0.65 * SUPERSAMPLE), 0, 1) * (slope > 1e-10)
        ink = np.maximum(0.4 * nodal_ink, 0.78 * self.rim)
        colours *= 1 - ink[:, None]

        size = PIXELS * SUPERSAMPLE
        picture = np.ones((size * size, 3))
        picture[self.pixel] = colours
        picture = picture.reshape(size, size, 3)[::-1]
        picture = picture.reshape(PIXELS, SUPERSAMPLE, PIXELS, SUPERSAMPLE, 3).mean(axis=(1, 3))
        return np.rint(255 * picture).astype(np.uint8)

    def outline(self, picture: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """The picture with the outlines of triangles `[tiles, 3, 2]` drawn over it."""
        size = PIXELS * SUPERSAMPLE
        screen = ((corners - self.centre) * self.scale + size / 2) / SUPERSAMPLE
        screen[..., 1] = PIXELS - screen[..., 1]
        image = Image.fromarray(picture)
        draw = ImageDraw.Draw(image)
        for triangle in screen:
            draw.polygon([tuple(point) for point in triangle], outline=OUTLINE, width=2)
        return np.asarray(image)


def frames(
    vertices: tuple[Vector, Vector], triangles: tuple[np.ndarray, np.ndarray], tiles: Vector,
    fields: Iterable[tuple[Scalar, Scalar]], exposure: float,
) -> list[np.ndarray]:
    """Two flat displacement maps sharing a fixed scale and colour range, each with the outlines of its
    seven triangles `tiles [2, tiles, 3]`."""
    points = tuple(coordinates(value) for value in vertices)
    corners = coordinates(tiles)                                             # [2, tiles, 3, 2]
    scale = 0.91 * PIXELS * SUPERSAMPLE / max(np.ptp(value, axis=0).max() for value in points)
    membranes = tuple(MembraneRaster(value, faces, scale) for value, faces in zip(points, triangles))
    return [np.concatenate([membrane.outline(membrane.draw(amplitudes(field), exposure), drum)
                            for membrane, field, drum in zip(membranes, pair, corners)], axis=1)
            for pair in fields]


def draw_pieces(vertices: Vector, triangles: np.ndarray, pieces: Scalar, exposure: float) -> plt.Figure:
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
