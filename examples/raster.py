"""Rasterising triangle meshes into images, as a renderer's shader sees them: for every pixel, the nearest
triangle covering it, the pixel centre's barycentric weights in that triangle, and its depth. Anything
given at the corners of the triangles, a colour, a normal, a phase, interpolates per pixel from these."""

from __future__ import annotations

import numpy as np

# Triangles rasterised together, bounding the memory one pass takes.
CHUNK = 4096


def view_frame(elevation: float, azimuth: float) -> np.ndarray:
    """The rows screen right, screen up and toward the viewer `[3, 3]`, for a view from the given elevation
    and azimuth in degrees, with z up."""
    elevation, azimuth = np.radians(elevation), np.radians(azimuth)
    toward = np.array([np.cos(elevation) * np.cos(azimuth), np.cos(elevation) * np.sin(azimuth), np.sin(elevation)])
    right = np.cross([0.0, 0.0, 1.0], toward)
    right /= np.linalg.norm(right)
    return np.stack([right, np.cross(toward, right), toward])


def screen(points: np.ndarray, frame: np.ndarray, centre: np.ndarray, reach: float, size: int) -> np.ndarray:
    """Points `[..., 3]` in pixel coordinates and depth toward the viewer, for an image of the given side
    in pixels showing the given reach around the centre."""
    viewed = (points - centre) @ frame.T
    return viewed * (size / (2 * reach)) + [size / 2, size / 2, 0.0]


def rasterise(screen: np.ndarray, faces: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The nearest triangle at each covered pixel, with the pixel centre's barycentric weights in it.

    `screen` holds each vertex's pixel coordinates and its depth toward the viewer. Returns the flat pixel
    indices, the triangle covering each, the weights of its three corners, and the depth.
    """
    pixels, owners, depths, weights = [], [], [], []
    for start in range(0, len(faces), CHUNK):
        chunk = np.arange(start, min(start + CHUNK, len(faces)))
        corners = screen[faces[chunk]]                                       # [T, 3, 3]
        # the pixels whose centres fall within each triangle's bounds
        low = np.clip(np.ceil(corners[..., :2].min(axis=1) - 0.5), 0, size - 1).astype(int)
        high = np.clip(np.floor(corners[..., :2].max(axis=1) - 0.5), 0, size - 1).astype(int)
        span = np.maximum(high - low + 1, 0)
        counts = span[:, 0] * span[:, 1]
        owner = np.repeat(np.arange(len(chunk)), counts)
        local = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
        x = low[owner, 0] + local % span[owner, 0]
        y = low[owner, 1] + local // span[owner, 0]
        a, b, c = (corners[owner, k] for k in range(3))
        area = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        centre_x, centre_y = x + 0.5, y + 0.5
        first = ((b[:, 0] - centre_x) * (c[:, 1] - centre_y) - (b[:, 1] - centre_y) * (c[:, 0] - centre_x)) / area
        second = ((c[:, 0] - centre_x) * (a[:, 1] - centre_y) - (c[:, 1] - centre_y) * (a[:, 0] - centre_x)) / area
        weight = np.stack([first, second, 1 - first - second], axis=-1)
        inside = (weight >= 0).all(axis=-1) & (area != 0)
        pixels.append((y * size + x)[inside])
        owners.append(chunk[owner[inside]])
        depths.append(np.einsum("pk,pk->p", weight[inside], np.stack([a[inside, 2], b[inside, 2], c[inside, 2]], -1)))
        weights.append(weight[inside])
    pixel, owner, depth, weight = (np.concatenate(parts) for parts in (pixels, owners, depths, weights))
    order = np.lexsort((-depth, pixel))
    nearest = order[np.r_[True, pixel[order][1:] != pixel[order][:-1]]]
    return pixel[nearest], owner[nearest], weight[nearest], depth[nearest]
