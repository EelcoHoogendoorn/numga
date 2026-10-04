"""Drawing for spin transformations: shaded meshes coloured by a field on their faces, and surfaces
rendered with a z-buffer, smoothly lit on both sides and textured with a checkerboard carried over
from the sphere they came from."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from examples.animation import capture
from examples.geometry.spin_transformations.core import Mesh, Scalar, ga

# The direction the light comes from, and the share of a face's colour it cannot darken.
LIGHT = np.array([0.4, -0.5, 0.8]) / np.linalg.norm([0.4, -0.5, 0.8])
AMBIENT = 0.45


def euclidean(vertices) -> np.ndarray:
    return vertices.cast(ga.subspace("x y z")).kernel


def draw_mesh(ax, mesh: Mesh, colours: np.ndarray, title: str) -> None:
    """The mesh, each face in its colour, darkened as it turns from the light."""
    triangles = euclidean(mesh.vertices)[mesh.faces]                          # [F, 3, 3]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    light = AMBIENT + (1 - AMBIENT) * np.clip(normals @ LIGHT, 0, 1)
    shaded = colours.copy()
    shaded[:, :3] *= light[:, None]
    ax.add_collection3d(Poly3DCollection(triangles, facecolors=shaded, edgecolors=(0, 0, 0, 0.12), linewidths=0.3))
    extent = np.abs(triangles).max()
    ax.set_xlim(-extent, extent), ax.set_ylim(-extent, extent), ax.set_zlim(-extent, extent)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=18, azim=-60)
    ax.set_axis_off()
    ax.set_title(title)


def face_colours(values: Scalar, limit: float) -> np.ndarray:
    """A diverging colour per face for values in [-limit, limit]."""
    return plt.get_cmap("coolwarm")((values.to_array().ravel() / limit + 1) / 2)


def draw_surface(mesh: Mesh, values: Scalar, limit: float, title: str) -> plt.Figure:
    """The mesh, its faces coloured by a field in [-limit, limit]."""
    figure = plt.figure(figsize=(5, 5))
    draw_mesh(figure.add_subplot(projection="3d"), mesh, face_colours(values, limit), title)
    figure.tight_layout()
    return figure


def draw_dipole(mesh: Mesh, curvature: Scalar, deformed: Mesh) -> plt.Figure:
    """The sphere coloured by the curvature change it is given, and the surface that has it,
    textured with the sphere's checkerboard."""
    limit = np.abs(curvature.to_array()).max()
    figure = plt.figure(figsize=(10, 5))
    draw_mesh(figure.add_subplot(1, 2, 1, projection="3d"), mesh, face_colours(curvature, limit), "change in mean curvature")
    axis = figure.add_subplot(1, 2, 2)
    axis.imshow(render_surface(deformed, mesh))
    axis.set_axis_off()
    axis.set_title("spin transformation")
    figure.tight_layout()
    return figure


def draw_textured(mesh: Mesh, reference: Mesh, title: str) -> plt.Figure:
    """The surface rendered with the reference's checkerboard."""
    figure, axis = plt.subplots(figsize=(5, 5))
    axis.imshow(render_surface(mesh, reference))
    axis.set_axis_off()
    axis.set_title(title)
    figure.tight_layout()
    return figure


def draw_dirac(gallery: dict[int, list[Mesh]], reference: Mesh) -> plt.Figure:
    """A row of Dirac spheres for each eigenvalue, textured with the sphere's checkerboard."""
    rows, columns = len(gallery), max(len(spheres) for spheres in gallery.values())
    figure, axes = plt.subplots(rows, columns, figsize=(3 * columns, 3 * rows), squeeze=False)
    for row, (eigenvalue, spheres) in enumerate(gallery.items()):
        for column, sphere in enumerate(spheres):
            axes[row, column].imshow(render_surface(sphere, reference))
            axes[row, column].set_title(f"n = {eigenvalue}")
        for axis in axes[row]:
            axis.set_axis_off()
    figure.tight_layout()
    return figure


def animate_rounding(frames: list[Mesh]) -> list[np.ndarray]:
    """Each step of the flow, textured with the checkerboard of the first."""
    images = []
    for step, frame in enumerate(frames):
        figure = draw_textured(frame, frames[0], f"conformal flow, step {step}")
        images.append(capture(figure))
        plt.close(figure)
    return images


# The camera's elevation and azimuth in degrees, the rendered image's side in pixels, and how many
# times finer each side is rasterised before averaging, against jagged edges.
ELEVATION, AZIMUTH = 20.0, -55.0
PIXELS = 420
SUPERSAMPLE = 2
# The checkerboard's cubes across the reference surface, the light and dark squares on the outer and inner
# side of the surface, and the strength and sharpness of the highlight.
SQUARES = 8
OUTSIDE = np.array([[0.98, 0.84, 0.62], [0.90, 0.45, 0.18]])
INSIDE = np.array([[0.70, 0.82, 0.96], [0.22, 0.42, 0.72]])
SPECULAR, SHININESS = 0.35, 40.0
# Triangles rasterised together, bounding the memory one pass takes.
CHUNK = 4096


def render_surface(mesh: Mesh, reference: Mesh) -> np.ndarray:
    """The surface as an RGB image, its texture a solid checkerboard of cubes sampled on the
    reference mesh with the same faces: straight squares on its flat faces, circles on a sphere."""
    points, original, faces = euclidean(mesh.vertices), euclidean(reference.vertices), mesh.faces
    triangles = points[faces]
    face_normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals = np.zeros_like(points)
    np.add.at(normals, faces, face_normals[:, None, :])
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)

    # The view frame: screen right, screen up, toward the viewer; the surface scaled to fill the image.
    elevation, azimuth = np.radians(ELEVATION), np.radians(AZIMUTH)
    toward = np.array([np.cos(elevation) * np.cos(azimuth), np.cos(elevation) * np.sin(azimuth), np.sin(elevation)])
    right = np.cross([0.0, 0.0, 1.0], toward)
    right /= np.linalg.norm(right)
    frame = np.stack([right, np.cross(toward, right), toward])
    size = PIXELS * SUPERSAMPLE
    viewed = (points - points.mean(axis=0)) @ frame.T
    scale = 0.46 * size / np.abs(viewed[:, :2]).max()
    screen = viewed * scale + [size / 2, size / 2, 0.0]

    pixel, face, weights = _rasterise(screen, faces, size)
    normal = np.einsum("pk,pkd->pd", weights, normals[faces[face]]) @ frame.T
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    position = np.einsum("pk,pkd->pd", weights, original[faces[face]])

    # The checkerboard: cubes centred on the origin, SQUARES of them across the reference surface.
    cells = np.rint(position / np.abs(original).max() * SQUARES / 2).astype(int)
    check = cells.sum(axis=-1) % 2
    facing = face_normals[face] @ toward > 0
    colour = np.where(facing[:, None], OUTSIDE[check], INSIDE[check])

    # Both sides lit alike: the normal turned toward the viewer, a diffuse term and a highlight.
    normal *= np.where(facing, 1.0, -1.0)[:, None]
    light = frame @ LIGHT
    halfway = (light + [0, 0, 1]) / np.linalg.norm(light + [0, 0, 1])
    diffuse = AMBIENT + (1 - AMBIENT) * np.clip(normal @ light, 0, 1)
    highlight = SPECULAR * np.clip(normal @ halfway, 0, 1) ** SHININESS
    shaded = np.clip(colour * diffuse[:, None] + highlight[:, None], 0, 1)

    image = np.ones((size * size, 3))
    image[pixel] = shaded
    image = image.reshape(size, size, 3)[::-1]
    return image.reshape(PIXELS, SUPERSAMPLE, PIXELS, SUPERSAMPLE, 3).mean(axis=(1, 3))


def _rasterise(screen: np.ndarray, faces: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The nearest triangle at each covered pixel, with the pixel centre's barycentric weights in it.

    `screen` holds each vertex's pixel coordinates and its depth toward the viewer. Returns the
    flat pixel indices, the triangle covering each, and the weights of its three corners.
    """
    pixels, owners, depths, weights = [], [], [], []
    for start in range(0, len(faces), CHUNK):
        chunk = np.arange(start, min(start + CHUNK, len(faces)))
        corners = screen[faces[chunk]]                                       # [T, 3, 3]
        low = np.clip(np.floor(corners[..., :2].min(axis=1)), 0, size - 1).astype(int)
        high = np.clip(np.ceil(corners[..., :2].max(axis=1)), 0, size - 1).astype(int)
        span = high - low + 1
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
    return pixel[nearest], owner[nearest], weight[nearest]
