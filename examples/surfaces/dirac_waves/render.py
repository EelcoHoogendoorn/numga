"""GPU surface rendering for spinor fields on a triangulated surface, mapping yz, zx, and xy amplitudes to RGB."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np

from numga import stack
from numga.render import meshes, solids
from numga.topology import Topology
from examples.surfaces.dirac_waves import core
from examples.util.panels import show_pixels

# First-frame exposure percentile, headroom multipliers for the focusing pulse and standing modes, surface grey, camera angles, and tile width.
exposure_percentile = 99.3
pulse_headroom, mode_headroom = 1.9, 1.0
surface_grey = 0.12
elevation, azimuth = np.radians(8.0), np.radians(35.0)
margin = 0.06
n_pixels = 192
background = np.zeros(3)

surface_colour = meshes.Colour(
    """
fn colour(values: vec4<f32>, face_values: vec4<f32>, corner: vec3<f32>, front: bool) -> vec4<f32> {
    return vec4<f32>(base + sqrt(values.rgb * values.rgb + face_values.rgb * face_values.rgb), 1.0);
}
""",
    dict(base=surface_grey),
)


def bivector_channels(fields: core.Even) -> np.ndarray:
    """Reads the absolute yz, zx, and xy plane amplitudes (`[..., 3]`) by pairing `fields` against `x`, `y`, and `z`."""
    mv = core.mv
    return np.abs((stack((mv.x, mv.y, mv.z)) & fields.batch()[..., None]).to_array())


def tile_image(topology: Topology, camera_scene: tuple, corner_rgb: np.ndarray, face_rgb: np.ndarray) -> np.ndarray:
    """Renders one lit surface tile (`[n_pixels, n_pixels, 3]`) blending corner vertex colours (`[3, F, 3]`) with triangle colours (`[F, 3]`)."""
    points, normals, frame = camera_scene
    return (
        meshes.draw(
            points,
            topology.elements[2],
            corner_rgb.transpose(1, 0, 2),
            normals,
            frame,
            background,
            face_values=face_rgb,
            colour=surface_colour,
            gloss=solids.gloss,
        )
        / 255
    )


def surface_camera(topology: Topology, vertices: core.Point) -> tuple:
    """Builds the projective vertex points (`[V] Point-`), smooth normals, and camera framing for `vertices`."""
    points = solids.origin + solids.context.include_from_algebra(vertices.batch())   # [V] Point-
    frame = solids.view([points], 0.0, elevation, azimuth, margin, n_pixels)
    return points, meshes.smooth(points, topology), frame.squared()


def tiled_animation(
    topology: Topology,
    vertices: core.Point,
    n_columns: int,
    headroom: float,
    states: Iterable[core.SpinorPair],
) -> Iterator[np.ndarray]:
    """Streams `(vertex_spinor, face_spinor)` states into tiled RGB frames (`Iterator[np.ndarray]`), calibrating exposure from the first frame."""
    state_iter = iter(states)
    first_vertex, first_face = next(state_iter)
    # Normalize single-tile fields (`Even+[V]`) and multi-tile batches (`[n_tiles] Even+[V]`) to leading `[n_tiles]`.
    is_single = first_vertex.ndim == 0
    if is_single:
        first_vertex, first_face = first_vertex[None], first_face[None]

    first_v_rgb = bivector_channels(first_vertex)                               # [..., V, 3]
    first_f_rgb = bivector_channels(first_face)                                 # [..., F, 3]
    combined = np.concatenate([first_v_rgb, first_f_rgb], axis=1)               # [..., V + F, 3]
    scale = np.maximum(headroom * np.percentile(combined, exposure_percentile, axis=(1, 2), keepdims=True), 1e-12)
    n_tiles = first_v_rgb.shape[0]
    n_rows = (n_tiles + n_columns - 1) // n_columns
    camera_scene = surface_camera(topology, vertices)

    def render_frame(v_field: core.Even, f_field: core.Even) -> np.ndarray:
        if is_single:
            v_field, f_field = v_field[None], f_field[None]
        corner_rgb = bivector_channels(topology.incidence[2, 0] * v_field[..., None]) / scale[..., None]   # [..., 3, F, 3]
        face_rgb = bivector_channels(f_field) / scale                           # [..., F, 3]
        pictures = np.stack([
            tile_image(topology, camera_scene, corner_rgb[tile], face_rgb[tile]) for tile in range(n_tiles)
        ])
        padded = np.concatenate([pictures, np.zeros((-n_tiles % n_columns,) + pictures.shape[1:])])
        grid = padded.reshape((n_rows, n_columns, n_pixels, n_pixels, 3))
        canvas = grid.transpose(0, 2, 1, 3, 4).reshape(n_rows * n_pixels, n_columns * n_pixels, 3)
        return np.round(255 * canvas).astype(np.uint8)

    yield render_frame(first_vertex[0] if is_single else first_vertex, first_face[0] if is_single else first_face)
    for vertex_spinor, face_spinor in state_iter:
        yield render_frame(vertex_spinor, face_spinor)


def pulse_animation(
    topology: Topology, vertices: core.Point, states: Iterable[core.SpinorPair],
) -> Iterator[np.ndarray]:
    """Streams a single-sphere spinor pulse (`Even+[V]`, `Even-[F]`) with `pulse_headroom` exposure."""
    return tiled_animation(topology, vertices, 1, pulse_headroom, states)


def modes_animation(
    topology: Topology, vertices: core.Point, n_columns: int, states: Iterable[core.SpinorPair],
) -> Iterator[np.ndarray]:
    """Streams `n_columns` standing-wave modes (`[n_columns] Even+[V]`, `[n_columns] Even-[F]`) side by side."""
    return tiled_animation(topology, vertices, n_columns, mode_headroom, states)


def spinor_figure(topology: Topology, vertices: core.Point, vertex_spinor: core.Even, face_spinor: core.Even) -> plt.Figure:
    """Draws a static figure of `vertex_spinor` (`Even+[V]`) and `face_spinor` (`Even-[F]`) on the triangulated surface."""
    (picture,) = tiled_animation(topology, vertices, 1, mode_headroom, [(vertex_spinor, face_spinor)])
    figure, ax = plt.subplots(figsize=(3, 3))
    show_pixels(ax, picture)
    return figure
