"""Depth-tested views of saddle surfaces and their tangent partners."""

from __future__ import annotations

from collections.abc import Iterable
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np

from numga import stack
from examples.raster import rasterise, view_frame
from examples.surfaces.k_surfaces import core

n_width = 640
n_height = 480
VIEW = (15, -15)
LIGHT = np.array([0.3, -0.5, 0.8])
FRONT = np.array([0.29, 0.61, 0.67])
BACK = np.array([0.79, 0.48, 0.32])
AMBIENT = 0.44
EDGE_WIDTH = 0.8
EDGE_DARKNESS = 0.28
BOUNDARY_DARKNESS = 0.6
SOURCE = np.array([0.28, 0.58, 0.64])
PARTNER = np.array([0.78, 0.48, 0.28])
SURFACE_OPACITY = 0.78
PAIR_GRID_INK = 0.6
CONNECTOR = np.array([0.17, 0.17, 0.20])
CONNECTOR_WIDTH = 1.8
ENDPOINT_RADIUS = 2.5
HIDDEN_OPACITY = 0.35
n_connector_rows = 3
n_connector_columns = 3
DEPTH_BIAS = 0.6


# --- plumbing

def coordinates(points: core.Point) -> np.ndarray:
    """Positions read by pairing with the planes across the axes."""
    mv = points.context.multivector
    return (stack((mv.x, mv.y, mv.z)) & points[..., None]).to_array()


def mesh(n_rows: int, n_columns: int) -> tuple[np.ndarray, np.ndarray]:
    """Triangles and their edge ink, with the diagonal of each quad left unmarked."""
    indices = np.arange(n_rows * n_columns).reshape(n_rows, n_columns)
    quads = np.stack([indices[:-1, :-1], indices[1:, :-1],
                     indices[1:, 1:], indices[:-1, 1:]], axis=-1).reshape(-1, 4)
    faces = quads[:, [[0, 1, 2], [0, 2, 3]]].reshape(-1, 3)
    edge_vertices = np.sort(faces[:, [[1, 2], [2, 0], [0, 1]]], axis=-1)
    _, inverse, counts = np.unique(edge_vertices.reshape(-1, 2), axis=0,
                                  return_inverse=True, return_counts=True)
    boundary = counts[inverse].reshape(-1, 3) == 1
    grid_edges = np.tile([[1, 0, 1], [1, 1, 0]], (len(quads), 1))
    ink = np.where(boundary, BOUNDARY_DARKNESS, EDGE_DARKNESS) * grid_edges
    return faces, ink


def viewport(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """One orthographic camera and scale for an entire deformation."""
    camera = view_frame(*VIEW)
    projected = points.reshape(-1, 3) @ camera.T
    centre = (projected.min(axis=0) + projected.max(axis=0)) / 2
    scale = 0.82 * np.min(np.array([n_width, n_height]) / np.ptp(projected[:, :2], axis=0))
    return camera, centre, scale


def surface(points: np.ndarray, faces: np.ndarray, edge_ink: np.ndarray,
            camera: np.ndarray, centre: np.ndarray, scale: float,
            front: np.ndarray, back: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Smooth two-sided shading and quad lines sharing one depth buffer."""
    points = points.reshape(-1, 3)
    screen = (points @ camera.T - centre) * scale + [n_width / 2, n_width / 2, 0]
    pixel, face, weights, visible_depth = rasterise(screen, faces, n_width)

    triangles = points[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0],
                       triangles[:, 2] - triangles[:, 0])
    vertex_normals = np.zeros_like(points)
    np.add.at(vertex_normals, faces, normals[:, None])
    vertex_normals /= np.linalg.norm(vertex_normals, axis=-1, keepdims=True)
    normal = np.einsum("pk,pkd->pd", weights, vertex_normals[faces[face]])
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    # Winding fixes each face's material; interpolated normals only control its lighting.
    facing = (normals @ camera[2] > 0)[face]
    colour = np.where(facing[:, None], front, back)
    outward = normal * np.where(facing, 1, -1)[:, None]
    light = LIGHT / np.linalg.norm(LIGHT)
    diffuse = AMBIENT + (1 - AMBIENT) * np.maximum(outward @ light, 0)
    linear = np.where(colour <= 0.04045, colour / 12.92, ((colour + 0.055) / 1.055) ** 2.4)
    linear *= diffuse[:, None]

    # Distances within the visible triangle keep hidden grid lines behind the surface.
    corners = screen[faces, :2]
    edges = np.roll(corners, -1, axis=1) - np.roll(corners, 1, axis=1)
    lengths = np.linalg.norm(edges, axis=-1)
    first, second = corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]
    area = np.abs(first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0])
    distance = weights * (area[:, None] / lengths)[face]
    ink = np.max(np.clip(1 - distance / EDGE_WIDTH, 0, 1) * edge_ink[face], axis=-1)
    linear *= 1 - ink[:, None]
    colour = np.where(linear <= 0.0031308, linear * 12.92, 1.055 * linear ** (1 / 2.4) - 0.055)

    image = np.ones((n_width * n_width, 3))
    image[pixel] = colour
    depth = np.full(n_width * n_width, -np.inf)
    depth[pixel] = visible_depth
    return image, depth


def pixels(image: np.ndarray) -> np.ndarray:
    """Crop the square raster to its landscape canvas and convert to RGB bytes."""
    image = image.reshape(n_width, n_width, 3)[::-1]
    margin = (n_width - n_height) // 2
    return np.rint(image[margin:margin + n_height] * 255).astype(np.uint8)


def picture(points: np.ndarray, faces: np.ndarray, edge_ink: np.ndarray,
            camera: np.ndarray, centre: np.ndarray, scale: float) -> np.ndarray:
    image, _ = surface(points, faces, edge_ink, camera, centre, scale, FRONT, BACK)
    return pixels(image)


def lines(image: np.ndarray, depth: np.ndarray, segments: np.ndarray,
          colour: np.ndarray, width: float) -> None:
    """Antialiased strokes, clipped against the surface depth at every pixel."""
    first, delta = segments[:, 0], segments[:, 1] - segments[:, 0]
    counts = np.ceil(np.max(np.abs(delta[:, :2]), axis=-1)).astype(int) * 2 + 2
    owner = np.repeat(np.arange(len(segments)), counts)
    local = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
    samples = first[owner] + delta[owner] * (local / (counts[owner] - 1))[:, None]
    offsets = np.array([[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 0],
                        [0, 1], [1, -1], [1, 0], [1, 1]])
    positions = (np.floor(samples[:, None, :2]).astype(int) + offsets).reshape(-1, 2)
    owner = np.repeat(owner, len(offsets))
    centres = positions + 0.5
    fraction = np.clip(np.einsum("pi,pi->p", centres - first[owner, :2], delta[owner, :2])
                       / np.sum(delta[owner, :2] ** 2, axis=-1), 0, 1)
    nearest = first[owner] + delta[owner] * fraction[:, None]
    distance = np.linalg.norm(centres - nearest[:, :2], axis=-1)
    coverage = np.clip(width / 2 + 0.5 - distance, 0, 1)
    pixel = np.clip(positions[:, 1] * n_width + positions[:, 0], 0, len(depth) - 1)
    visible = ((positions >= 0) & (positions < n_width)).all(axis=-1)
    visible &= nearest[:, 2] + DEPTH_BIAS >= depth[pixel]
    opacity = np.zeros_like(depth)
    np.maximum.at(opacity, pixel[visible], coverage[visible])
    pixel = np.flatnonzero(opacity)
    image[pixel] = image[pixel] * (1 - opacity[pixel, None]) + colour * opacity[pixel, None]


def dots(image: np.ndarray, depth: np.ndarray, points: np.ndarray,
         colour: np.ndarray, radius: float) -> None:
    """Antialiased endpoint disks sharing the surfaces' depth buffer."""
    extent = int(np.ceil(radius))
    offsets = np.stack(np.meshgrid(np.arange(-extent, extent + 1),
                                   np.arange(-extent, extent + 1)), axis=-1).reshape(-1, 2)
    positions = (np.floor(points[:, None, :2]).astype(int) + offsets).reshape(-1, 2)
    owner = np.repeat(np.arange(len(points)), len(offsets))
    distance = np.linalg.norm(positions + 0.5 - points[owner, :2], axis=-1)
    coverage = np.clip(radius + 0.5 - distance, 0, 1)
    pixel = np.clip(positions[:, 1] * n_width + positions[:, 0], 0, len(depth) - 1)
    visible = ((positions >= 0) & (positions < n_width)).all(axis=-1)
    visible &= points[owner, 2] + DEPTH_BIAS >= depth[pixel]
    opacity = np.zeros_like(depth)
    np.maximum.at(opacity, pixel[visible], coverage[visible])
    pixel = np.flatnonzero(opacity)
    image[pixel] = image[pixel] * (1 - opacity[pixel, None]) + colour * opacity[pixel, None]


def partner_picture(partner: np.ndarray, source: np.ndarray, faces: np.ndarray,
                    edge_ink: np.ndarray, camera: np.ndarray, centre: np.ndarray,
                    scale: float) -> np.ndarray:
    """Two translucent sheets with common tangents and marked corresponding points."""
    source_image, source_depth = surface(source, faces, edge_ink * PAIR_GRID_INK,
                                         camera, centre, scale, SOURCE, SOURCE)
    partner_image, partner_depth = surface(partner, faces, edge_ink * PAIR_GRID_INK,
                                           camera, centre, scale, PARTNER, PARTNER)
    images = np.stack([source_image, partner_image])
    depths = np.stack([source_depth, partner_depth])
    order = np.argsort(depths, axis=0)
    layers = np.take_along_axis(images, order[..., None], axis=0)
    opacity = SURFACE_OPACITY * np.isfinite(np.take_along_axis(depths, order, axis=0))
    image = np.ones_like(source_image)
    for layer, alpha in zip(layers, opacity):
        image = image * (1 - alpha[:, None]) + layer * alpha[:, None]
    depth = np.maximum(source_depth, partner_depth)

    source_screen = (source @ camera.T - centre) * scale + [n_width / 2, n_width / 2, 0]
    partner_screen = (partner @ camera.T - centre) * scale + [n_width / 2, n_width / 2, 0]
    n_rows, n_columns = source.shape[:2]
    row_margin, column_margin = n_rows // 6, n_columns // 6
    rows = np.linspace(row_margin, n_rows - row_margin - 1, n_connector_rows, dtype=int)
    columns = np.linspace(column_margin, n_columns - column_margin - 1, n_connector_columns, dtype=int)
    connectors = np.stack([source_screen[rows[:, None], columns],
                           partner_screen[rows[:, None], columns]], axis=-2).reshape(-1, 2, 3)
    endpoints = connectors.reshape(-1, 3)
    # Faint hidden segments keep the correspondence legible through overlapping sheets.
    underlay = image.copy()
    free_depth = np.full_like(depth, -np.inf)
    lines(underlay, free_depth, connectors, CONNECTOR, CONNECTOR_WIDTH)
    dots(underlay, free_depth, endpoints, CONNECTOR, ENDPOINT_RADIUS)
    image = image * (1 - HIDDEN_OPACITY) + underlay * HIDDEN_OPACITY
    lines(image, depth, connectors, CONNECTOR, CONNECTOR_WIDTH)
    dots(image, depth, endpoints, CONNECTOR, ENDPOINT_RADIUS)
    return pixels(image)


def frames(states: Iterable[core.Point], reference: core.Point) -> list[np.ndarray]:
    """Draw each point grid with the camera fixed by a reference grid."""
    states = iter(states)
    first = coordinates(next(states))
    faces, edge_ink = mesh(*first.shape[:2])
    camera, centre, scale = viewport(coordinates(reference))
    positions = chain([first], (coordinates(state) for state in states))
    return [picture(points, faces, edge_ink, camera, centre, scale) for points in positions]


def partner_frames(states: Iterable[core.Point], source: core.Point,
                   reference: core.Point) -> list[np.ndarray]:
    """Draw a partner family against one source net and one fixed camera."""
    source = coordinates(source)
    faces, edge_ink = mesh(*source.shape[:2])
    camera, centre, scale = viewport(coordinates(reference))
    return [partner_picture(coordinates(state), source, faces, edge_ink, camera, centre, scale)
            for state in states]


def partner_orbit(partner: core.Point, source: core.Point,
                  azimuths: np.ndarray, elevation: float) -> list[np.ndarray]:
    """Orbit a fixed pair with one centre and scale fitted to all projected views."""
    source, partner = coordinates(source), coordinates(partner)
    faces, edge_ink = mesh(*source.shape[:2])
    cameras = np.stack([view_frame(elevation, azimuth) for azimuth in azimuths])
    points = np.concatenate([source.reshape(-1, 3), partner.reshape(-1, 3)])
    centre = (points.min(axis=0) + points.max(axis=0)) / 2
    projected = np.einsum("ni,fji->fnj", points - centre, cameras)
    span = 2 * np.abs(projected[..., :2]).max(axis=(0, 1))
    scale = 0.9 * np.min(np.array([n_width, n_height]) / span)
    return [partner_picture(partner, source, faces, edge_ink, camera, centre @ camera.T, scale)
            for camera in cameras]


def image_figure(image: np.ndarray) -> plt.Figure:
    """An image at its pixel size, without axes or margins."""
    figure, ax = plt.subplots(figsize=(n_width / 100, n_height / 100), dpi=100)
    ax.imshow(image)
    ax.set_axis_off()
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
    return figure


def draw(points: core.Point, reference: core.Point) -> plt.Figure:
    """One surface filling its canvas, without axes or margins."""
    image, = frames([points], reference)
    return image_figure(image)


def draw_partner(partner: core.Point, source: core.Point, reference: core.Point) -> plt.Figure:
    """One partner with its source surface and common tangent segments."""
    image, = partner_frames([partner], source, reference)
    return image_figure(image)
