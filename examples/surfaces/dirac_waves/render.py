"""Drawing for spinor waves on a surface: each tile the surface over time, lit from above, coloured red,
green and blue by the planes yz, zx and xy of the vertex field together with those of the face field."""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.pyplot as plt
import numpy as np

from examples.surfaces.dirac_waves import core
from examples.util.wgsl import meshes, show_pixels, solids

Mesh = core.Mesh

# Each tile's exposure, the percentile of its first frame's amplitudes that shows at full brightness, and how many
# times that the pulse and the standing waves show at full brightness, room for the pulse as it gathers again; the
# surface's own grey under the field; the view; and each tile's pixels across.
exposure = 99.3
pulse_room, mode_room = 1.9, 1.0
base = 0.12
elevation, azimuth = np.radians(8.0), np.radians(35.0)
margin = 0.06
pixels = 192
background = np.zeros(3)

# A pixel's colour: the surface's grey, and the vertices' field blended across the face joined with the face's own.
colour = f"""
fn colour(values: vec4<f32>, face_values: vec4<f32>, corner: vec3<f32>, front: bool) -> vec4<f32> {{
    return vec4<f32>({base} + sqrt(values.rgb * values.rgb + face_values.rgb * face_values.rgb), 1.0);
}}
"""


def coordinates(points: core.Point) -> np.ndarray:
    """The x, y and z coordinates of points, antivectors."""
    return points.cast(core.ga.subspace("yz zx xy")).kernel


def planes(fields: core.Even) -> np.ndarray:
    """The absolute amplitudes of fields' planes yz, zx and xy `[..., sites, 3]`."""
    return np.abs(fields.cast(core.ga.subspace("yz zx xy")).kernel)


def image(mesh: Mesh, seen: tuple, corners: np.ndarray, sides: np.ndarray) -> np.ndarray:
    """The surface as an RGB image, `[pixels, pixels, 3]`: its vertices' colours spread over the faces between
    them, joined with each face's own, and lit."""
    points, normals, frame = seen
    return meshes.draw(points, mesh.faces, corners[mesh.faces], normals, frame, background, face_values=sides,
                       colour=colour, gloss=solids.gloss) / 255


def view(mesh: Mesh) -> tuple:
    """The surface as the camera sees it, the same for every field on it: its points carried into projective
    space, their smooth normals, and the camera framing them."""
    points = solids.mv.point(coordinates(mesh.vertices))
    frame = solids.view([points], 0.0, elevation, azimuth, margin, pixels)
    return points, meshes.smooth(points, mesh.faces), frame.squared()


def animate(mesh: Mesh, vertices: core.Even, faces: core.Even, columns: int,
            room: float = 1.0) -> Iterator[np.ndarray]:
    """Frames of vertex and face fields `[tiles, frames] Even[V]` and `[tiles, frames] Even[F]`, the tiles in rows
    of the given count, each exposed by its first frame, `room` times over."""
    corners, sides = planes(vertices), planes(faces)                          # [tiles, frames, V, 3], [tiles, frames, F, 3]
    first = np.concatenate([corners[:, :1], sides[:, :1]], axis=2)            # [tiles, 1, V + F, 3]
    level = room * np.percentile(first, exposure, axis=(1, 2, 3), keepdims=True)
    corners, sides = corners / level, sides / level
    tiles, frames = corners.shape[:2]
    seen = view(mesh)
    for frame in range(frames):
        pictures = np.stack([image(mesh, seen, corners[tile, frame], sides[tile, frame]) for tile in range(tiles)])
        pictures = np.concatenate([pictures, np.zeros((-tiles % columns,) + pictures.shape[1:])])
        rows = pictures.reshape((-1, columns) + pictures.shape[1:])           # [rows, columns, P, P, 3]
        layout = rows.transpose(0, 2, 1, 3, 4).reshape(rows.shape[0] * pixels, columns * pixels, 3)
        yield np.round(255 * layout).astype(np.uint8)


def animate_pulse(mesh: Mesh, vertices: core.Even, faces: core.Even,
                  room: float = pulse_room) -> Iterator[np.ndarray]:
    """Frames of a wave's vertex and face fields `[frames] Even[V]` and `[frames] Even[F]`, exposed by its first frame,
    `room` times over."""
    return animate(mesh, vertices[None], faces[None], 1, room)


def animate_modes(mesh: Mesh, vertices: core.Even, faces: core.Even,
                  room: float = mode_room) -> Iterator[np.ndarray]:
    """Frames of standing waves' vertex and face fields `[waves, frames] Even[V]` and `[waves, frames] Even[F]`, side
    by side, each exposed by its first frame, `room` times over."""
    return animate(mesh, vertices, faces, vertices.shape[0], room)


def still(mesh: Mesh, vertices: core.Even, faces: core.Even) -> plt.Figure:
    """A vertex field `Even[V]` and a face field `Even[F]` on the surface, exposed together."""
    picture, = animate(mesh, vertices[None, None], faces[None, None], 1)
    figure, ax = plt.subplots(figsize=(3, 3))
    show_pixels(ax, picture)
    return figure
