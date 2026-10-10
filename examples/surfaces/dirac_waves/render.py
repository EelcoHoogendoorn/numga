"""Drawing for spinor waves on a surface: each tile the surface over time, lit from above, coloured red,
green and blue by the vertex field's planes yz, zx and xy together with the face field's directions x,
y and z across them."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.mesh import Mesh
from examples.surfaces.spin_transformations.render import _rasterise
from examples.surfaces.dirac_waves import core

# Each tile's exposure, the percentile of its amplitudes over all its frames that shows at full
# brightness; the surface's own grey under the field; the view; the light; and the image size.
EXPOSURE = 99.3
BASE = 0.12
ELEVATION, AZIMUTH = 8.0, -55.0
LIGHT = np.array([0.4, -0.5, 0.8]) / np.linalg.norm([0.4, -0.5, 0.8])
AMBIENT = 0.35
PIXELS = 128
SUPERSAMPLE = 2


def coordinates(points: core.Vector) -> np.ndarray:
    """The x, y and z coordinates of points."""
    return points.cast(core.ga.subspace("x y z")).kernel


def planes(fields: core.Even) -> np.ndarray:
    """The absolute amplitudes of vertex fields' planes yz, zx and xy `[..., V, 3]`."""
    return np.abs(fields.cast(core.ga.subspace("yz zx xy")).kernel)


def directions(fields: core.Odd) -> np.ndarray:
    """The absolute amplitudes of face fields' directions x, y and z, across those planes `[..., F, 3]`."""
    return np.abs(fields.cast(core.ga.subspace("x y z")).kernel)


def view(mesh: Mesh) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The surface as the camera sees it, the same for every field on it: the covered pixels of the
    supersampled image, the face at each and its corners' weights there, and the light falling there."""
    points, triangles = coordinates(mesh.vertices), mesh.faces
    normals = coordinates(mesh.vertex_normals)
    elevation, azimuth = np.radians(ELEVATION), np.radians(AZIMUTH)
    toward = np.array([np.cos(elevation) * np.cos(azimuth), np.cos(elevation) * np.sin(azimuth), np.sin(elevation)])
    right = np.cross([0.0, 0.0, 1.0], toward)
    right /= np.linalg.norm(right)
    frame = np.stack([right, np.cross(toward, right), toward])
    size = PIXELS * SUPERSAMPLE
    viewed = points @ frame.T
    screen = viewed * (0.46 * size / np.abs(viewed[:, :2]).max()) + [size / 2, size / 2, 0.0]
    pixel, face, weights = _rasterise(screen, triangles, size)
    normal = np.einsum("pk,pkd->pd", weights, normals[triangles[face]]) @ frame.T
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    light = AMBIENT + (1 - AMBIENT) * np.clip(normal @ (frame @ LIGHT), 0, 1)
    return pixel, face, weights, light


def image(mesh: Mesh, seen: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], corners: np.ndarray,
          faces: np.ndarray) -> np.ndarray:
    """The surface as an RGB image: its vertices' colours spread over the faces between them, joined
    with each face's own, and lit."""
    pixel, face, weights, light = seen
    size = PIXELS * SUPERSAMPLE
    colour = np.hypot(np.einsum("pk,pkc->pc", weights, corners[mesh.faces[face]]), faces[face])
    picture = np.zeros((size * size, 3))
    picture[pixel] = np.clip((BASE + colour) * light[:, None], 0, 1)
    picture = picture.reshape(size, size, 3)[::-1]
    return picture.reshape(PIXELS, SUPERSAMPLE, PIXELS, SUPERSAMPLE, 3).mean(axis=(1, 3))


def animate(mesh: Mesh, vertices: core.Even, faces: core.Odd, columns: int) -> list[np.ndarray]:
    """Frames of vertex and face fields `[tiles, frames] Even[V]` and `[tiles, frames] Odd[F]`, the tiles in rows
    of the given count, each exposed over all its frames."""
    corners, sides = planes(vertices), directions(faces)                      # [tiles, frames, V, 3], [tiles, frames, F, 3]
    exposure = np.percentile(np.concatenate([corners, sides], axis=2), EXPOSURE, axis=(1, 2, 3), keepdims=True)
    corners, sides = corners / exposure, sides / exposure
    tiles, frames = corners.shape[:2]
    seen = view(mesh)
    pictures = np.stack([[image(mesh, seen, corners[tile, frame], sides[tile, frame]) for tile in range(tiles)]
                         for frame in range(frames)])
    pictures = np.concatenate([pictures, np.zeros((frames, -tiles % columns) + pictures.shape[2:])], axis=1)
    rows = pictures.reshape((frames, -1, columns) + pictures.shape[2:])       # [frames, rows, columns, P, P, 3]
    layout = rows.transpose(0, 1, 3, 2, 4, 5).reshape(frames, rows.shape[1] * PIXELS, columns * PIXELS, 3)
    return list((255 * layout).astype(np.uint8))


def still(mesh: Mesh, vertices: core.Even, faces: core.Odd) -> plt.Figure:
    """A vertex field `Even[V]` and a face field `Odd[F]` on the surface, exposed together."""
    picture, = animate(mesh, vertices[None, None], faces[None, None], 1)
    figure, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(picture)
    ax.set_axis_off()
    return figure
