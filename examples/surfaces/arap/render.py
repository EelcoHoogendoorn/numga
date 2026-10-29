"""Render the deformed bar meshes on the GPU with a rest-pose checkerboard pattern."""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np

from numga import stack
from examples.surfaces.arap import core
from examples.util.wgsl import image_figure, meshes, solids

# Checkerboard colours, camera elevation and azimuth in radians (azimuth zero looking along +y),
# bounding-box limits, viewport margin as a fraction of the largest box dimension, and panel width in pixels.
light, dark = np.array([0.98, 0.84, 0.62]), np.array([0.90, 0.45, 0.18])
elevation, azimuth = np.radians([18.0, 26.0])
limits = np.array([[-3.3, 3.5], [-1.5, 1.5], [-1.1, 4.0]])
margin = 0.06
n_pixels = 400
background = np.ones(3)


# --- plumbing -------------------------------------------------------------------------
def gpu_points(points: core.Point) -> solids.Point:
    """The GPU library's own points at the places of the example's: their coordinates, read by pairing the intrinsic
    points with the drawing frame, are the one hand-off from this algebra to the renderer's."""
    mv = core.mv
    return solids.mv.point((stack((mv.x, mv.y, mv.z)) & points.batch()[..., None]).to_array())


def checker(bar_mesh: core.Mesh, rest_vertices: core.Point, cell: np.ndarray) -> np.ndarray:
    """Per-face checkerboard colours from the parity of the grid cell containing each triangle's centroid at rest."""
    mv = core.mv
    centroids = bar_mesh.incidence[2, 0].mean(axis=0) * rest_vertices                # Point-[faces]
    # The planes of the checkerboard's grid, one cell apart: a point's pairings with them count the cells it lies across.
    cell_planes = stack((mv.x, mv.y, mv.z)) / cell                                  # [3] Vector+
    parity = np.floor((cell_planes & centroids.batch()[..., None]).to_array()).astype(int).sum(axis=-1) % 2
    return np.where(parity[:, None] == 1, dark, light)


def box_view(n_pixels: int) -> solids.View:
    """The camera view framing the fixed bounding box at `n_pixels` across."""
    corners = solids.mv.point(list(itertools.product(*limits)))                     # [corners] Point-
    return solids.view([corners], 0.0, elevation, azimuth, margin, n_pixels)


def picture(vertices: core.Point, faces: np.ndarray, colours: np.ndarray, view: solids.View) -> np.ndarray:
    """Render the bar's flat-shaded triangles with per-face checkerboard colours from the given view."""
    places = gpu_points(vertices)
    return meshes.draw(places, faces, np.zeros(3), meshes.flat(places, faces), view, background, face_values=colours,
                       colour=meshes.by_face, gloss=solids.gloss)


def side_by_side(bar_mesh: core.Mesh, rest_vertices: core.Point, shapes: list[core.Point],
                 cell: np.ndarray) -> plt.Figure:
    """Render the bar in each of the given shapes side by side, checkered by where each triangle sat at rest."""
    colours = checker(bar_mesh, rest_vertices, cell)
    view = box_view(n_pixels)
    return image_figure(np.concatenate([picture(shape, bar_mesh.faces, colours, view) for shape in shapes], axis=1))


def animate(bar_mesh: core.Mesh, rest_vertices: core.Point, shapes: Iterable[core.Point],
            cell: np.ndarray) -> Iterator[np.ndarray]:
    """Yield one rendered frame per shape of the bar, checkered by where each triangle sat at rest."""
    colours = checker(bar_mesh, rest_vertices, cell)
    view = box_view(n_pixels)
    for shape in shapes:
        yield picture(shape, bar_mesh.faces, colours, view)
