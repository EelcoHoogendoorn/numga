"""Smooth, depth-tested views of a deforming triangle surface."""

from __future__ import annotations

from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np

from numga import stack
from examples.raster import rasterise, view_frame
from examples.mechanics.freeform.geometry import Antivector

n_pixels = 460
n_supersamples = 2
VIEW = (24, -36)
LIGHT = np.array([-0.4, -0.7, 0.8])
SURFACE = np.array([0.25, 0.61, 0.68])
AMBIENT = 0.30


def coordinates(vertices: Antivector) -> np.ndarray:
    """Cartesian positions at the drawing boundary, read by pairing with the planes across the axes."""
    mv = vertices.context.multivector
    return (stack((mv.x, mv.y, mv.z)) & vertices[..., None]).to_array()


def viewport(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """A fixed three-quarter view with room for the surface to flex."""
    camera = view_frame(*VIEW)
    projected = points @ camera.T
    center = (projected.min(axis=0) + projected.max(axis=0)) / 2
    scale = 0.80 * n_pixels * n_supersamples / np.ptp(projected[:, :2], axis=0).max()
    return camera, center, scale


def picture(points: np.ndarray, faces: np.ndarray, camera: np.ndarray,
            center: np.ndarray, scale: float) -> np.ndarray:
    """Interpolate smooth surface normals through the visible triangle at each pixel."""
    size = n_pixels * n_supersamples
    screen = (points @ camera.T - center) * scale + [size / 2, size / 2, 0]
    pixel, face, weights, _ = rasterise(screen, faces, size)
    triangles = points[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    vertex_normals = np.zeros_like(points)
    np.add.at(vertex_normals, faces, normals[:, None])
    vertex_normals /= np.linalg.norm(vertex_normals, axis=-1, keepdims=True)
    normal = np.einsum("pk,pkd->pd", weights, vertex_normals[faces[face]])
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    light = LIGHT / np.linalg.norm(LIGHT)
    diffuse = AMBIENT + (1 - AMBIENT) * np.maximum(normal @ light, 0)
    halfway = light + camera[2]
    halfway /= np.linalg.norm(halfway)
    sheen = 0.08 * np.maximum(normal @ halfway, 0) ** 24
    linear = ((SURFACE + 0.055) / 1.055) ** 2.4
    colour = np.clip(linear * diffuse[:, None] + sheen[:, None], 0, 1)
    colour = np.where(colour <= 0.0031308, colour * 12.92, 1.055 * colour ** (1 / 2.4) - 0.055)
    image = np.ones((size * size, 3))
    image[pixel] = colour
    image = image.reshape(size, size, 3)[::-1]
    image = image.reshape(n_pixels, n_supersamples, n_pixels, n_supersamples, 3).mean(axis=(1, 3))
    return np.rint(image * 255).astype(np.uint8)


def frames(vertices: Iterable[Antivector], faces: np.ndarray) -> list[np.ndarray]:
    """Draw a sequence of surfaces with a camera fixed by the first frame."""
    states = iter(vertices)
    first = coordinates(next(states))
    camera, center, scale = viewport(first)
    images = [picture(first, faces, camera, center, scale)]
    images.extend(picture(coordinates(state), faces, camera, center, scale) for state in states)
    return images


def draw(vertices: Antivector, faces: np.ndarray) -> plt.Figure:
    """One surface without axes or margins."""
    points = coordinates(vertices)
    image = picture(points, faces, *viewport(points))
    figure, ax = plt.subplots(figsize=(n_pixels / 100, n_pixels / 100), dpi=100)
    ax.imshow(image)
    ax.set_axis_off()
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
    return figure
