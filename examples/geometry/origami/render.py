"""Depth-tested paper layers with two-sided diffuse lighting."""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.pyplot as plt
import numpy as np

from examples.geometry.origami import core

FRONT = np.array([0.90, 0.46, 0.30])
BACK = np.array([0.98, 0.88, 0.65])
EDGE = np.array([0.37, 0.24, 0.17])
PIXELS = 600
SUPERSAMPLE = 2
ELEVATION = 35
AZIMUTH = 110
LAYER_SPACING = 1e-6
LIGHT_DIRECTION = np.array([-0.4, 0.5, 1.0])
AMBIENT = 0.28


# --- plumbing -------------------------------------------------------------------------
def coordinates(points: core.Point) -> np.ndarray:
    values = points.cast(core.ga.subspace("yzw zxw xyw zyx")).kernel
    return values[..., :3] / values[..., 3:]


def camera() -> np.ndarray:
    elevation, azimuth = np.deg2rad([ELEVATION, AZIMUTH])
    eye = np.array([np.cos(elevation) * np.cos(azimuth),
                    np.cos(elevation) * np.sin(azimuth), np.sin(elevation)])
    right = np.array([-np.sin(azimuth), np.cos(azimuth), 0])
    return np.stack([right, np.cross(eye, right), eye])


def viewport(points: np.ndarray) -> tuple[np.ndarray, float]:
    viewed = points @ camera().T
    centre = (viewed.min(axis=0) + viewed.max(axis=0)) / 2
    scale = 0.88 * PIXELS * SUPERSAMPLE / np.max(np.ptp(viewed[:, :2], axis=0))
    return centre, scale


def rasterize(screen: np.ndarray, faces: tuple[np.ndarray, ...], colours: np.ndarray,
              bias: np.ndarray, size: int) -> np.ndarray:
    """Depth-test every pixel; a layer bias resolves coincident paper faces.

    Polygon-centre sorting cannot order overlapping coplanar polygons. Each triangle
    instead interpolates the true depth at the pixel, then adds its face's tiny bias.
    """
    depth = np.full((size, size), -np.inf)
    image = np.ones((size, size, 3))
    for face, colour, offset in zip(faces, colours, bias):
        # Convex paper faces triangulate as a fan; only their perimeter gets inked.
        for index in range(1, len(face) - 1):
            triangle = screen[face[[0, index, index + 1]]]
            first, second, third = triangle
            area = np.cross(second - first, third - first)[2]
            if abs(area) < 1e-10:
                continue
            low = np.maximum(np.floor(triangle[:, :2].min(axis=0)).astype(int), 0)
            high = np.minimum(np.ceil(triangle[:, :2].max(axis=0)).astype(int) + 1, size)
            y, x = np.mgrid[low[1]:high[1], low[0]:high[0]]
            x, y = x + 0.5, y + 0.5
            weight_first = ((second[0] - x) * (third[1] - y)
                            - (second[1] - y) * (third[0] - x)) / area
            weight_second = ((third[0] - x) * (first[1] - y)
                             - (third[1] - y) * (first[0] - x)) / area
            weight_third = 1 - weight_first - weight_second
            inside = (weight_first >= 0) & (weight_second >= 0) & (weight_third >= 0)
            distance = (weight_first * first[2] + weight_second * second[2]
                        + weight_third * third[2] + offset)
            region = np.s_[low[1]:high[1], low[0]:high[0]]
            visible = inside & (distance >= depth[region])
            depth[region][visible] = distance[visible]
            image[region][visible] = colour

    # Edges obey the same depth test, so covered creases do not shine through a flap.
    pen = np.array([(x, y) for x in range(-SUPERSAMPLE, SUPERSAMPLE + 1)
                    for y in range(-SUPERSAMPLE, SUPERSAMPLE + 1)
                    if x * x + y * y <= SUPERSAMPLE ** 2])
    for face, offset in zip(faces, bias):
        polygon = screen[face]
        normal = np.cross(polygon, np.roll(polygon, -1, axis=0)).sum(axis=0)
        for start, end in zip(screen[face], screen[np.roll(face, -1)]):
            samples = max(2, int(np.ceil(np.linalg.norm(end[:2] - start[:2]) * 2)))
            segment = start + np.linspace(0, 1, samples)[:, None] * (end - start)
            pixels = np.floor(segment[:, None, :2] + pen).astype(int)
            x, y = pixels[..., 0].reshape(-1), pixels[..., 1].reshape(-1)
            distance = np.repeat(segment[:, 2] + offset, len(pen))
            valid = (x >= 0) & (x < size) & (y >= 0) & (y < size)
            x, y, distance = x[valid], y[valid], distance[valid]
            # Test the face at the pixel centre, just as the fill does. Testing the
            # nearest point on a sloping edge would leak hidden outlines through it.
            if abs(normal[2]) > 1e-10:
                distance = (normal @ polygon[0] - normal[0] * (x + 0.5)
                            - normal[1] * (y + 0.5)) / normal[2] + offset
            visible = distance >= depth[y, x] - 1e-7
            image[y[visible], x[visible]] = EDGE
    return image


def picture(paper: core.Paper, centre: np.ndarray, scale: float) -> np.ndarray:
    points = coordinates(paper.points)
    offsets = paper.offsets.cast(core.ga.subspace("yzw zxw xyw")).kernel
    # Each small separation follows its facet's motor, including its sideways motion.
    # This preserves which material side is visible when a coincident stack starts moving.
    points = points + np.repeat(offsets, paper.counts, axis=0) * LAYER_SPACING
    basis = camera()
    screen = (points @ basis.T - centre) * scale
    size = PIXELS * SUPERSAMPLE
    screen[:, :2] += size / 2
    faces = paper.faces()
    normals = np.array([np.cross(points[face], points[np.roll(face, -1)]).sum(axis=0)
                        for face in faces])
    colours = shade(normals, basis[2])
    pixels = rasterize(screen, faces, colours, np.zeros(len(faces)), size)[::-1]
    pixels = pixels.reshape(PIXELS, SUPERSAMPLE, PIXELS, SUPERSAMPLE, 3).mean(axis=(1, 3))
    return np.rint(pixels * 255).astype(np.uint8)


def shade(normals: np.ndarray, eye: np.ndarray) -> np.ndarray:
    """Distinct front and back materials under fixed diffuse light and ambient fill."""
    normals = normals / np.linalg.norm(normals, axis=-1, keepdims=True)
    facing = normals @ eye > 0
    reflectance = np.where(facing[:, None], FRONT, BACK)
    outward = normals * np.where(facing, 1, -1)[:, None]
    light = LIGHT_DIRECTION / np.linalg.norm(LIGHT_DIRECTION)
    illumination = AMBIENT + (1 - AMBIENT) * np.maximum(outward @ light, 0)
    # Light scales linear reflectance; sRGB encoding happens only for the displayed colour.
    linear = np.where(reflectance <= 0.04045, reflectance / 12.92,
                      ((reflectance + 0.055) / 1.055) ** 2.4)
    lit = linear * illumination[:, None]
    return np.where(lit <= 0.0031308, lit * 12.92, 1.055 * lit ** (1 / 2.4) - 0.055)


def draw(paper: core.Paper) -> plt.Figure:
    centre, scale = viewport(coordinates(paper.points))
    figure, ax = plt.subplots(figsize=(6, 6), layout="constrained")
    ax.imshow(picture(paper, centre, scale))
    ax.set_axis_off()
    return figure


def animate(states: Iterator[core.Paper]) -> list[np.ndarray]:
    states = tuple(states)
    bounds = np.concatenate([coordinates(state.points) for state in states])
    centre, scale = viewport(bounds)
    return [picture(state, centre, scale) for state in states]
