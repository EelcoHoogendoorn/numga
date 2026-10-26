"""Drawing for as rigid as possible, on the GPU: the bar's deformations side by side, checkered by where each
triangle sat at rest."""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np

from examples.surfaces.arap import core
from examples.util.wgsl import fitted, meshes, show_pixels, solids

# The checkerboard's two colours, the view, elevation and azimuth in radians, azimuth zero looking along +y, the box
# it shows and the margin around it, as a share of its larger side, and the pixels across an animation's frame.
light, dark = np.array([0.98, 0.84, 0.62]), np.array([0.90, 0.45, 0.18])
elevation, azimuth = np.radians([18.0, 26.0])
limits = np.array([[-3.3, 3.5], [-1.5, 1.5], [-1.1, 4.0]])
margin = 0.06
n_pixels = 400
background = np.ones(3)


def coordinates(points: core.Point) -> np.ndarray:
    """The x, y and z coordinates of points."""
    return points.cast(core.ga.subspace("yz zx xy")).kernel


def checker(rest: core.Point, faces: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Each triangle's colour: the parity of the cell of the given size its centroid sat in at rest."""
    centroids = coordinates(rest)[faces].mean(axis=1)                          # [F, 3]
    parity = np.floor(centroids / cell).astype(int).sum(axis=-1) % 2
    return np.where(parity[:, None] == 1, dark, light)


def box_view(pixels: int) -> solids.View:
    """The view framing the fixed box, `pixels` across."""
    corners = solids.mv.point(np.array(list(itertools.product(*limits))))      # [8] Point
    return solids.view([corners], 0.0, elevation, azimuth, margin, pixels)


def picture(vertices: core.Point, faces: np.ndarray, colours: np.ndarray, view: solids.View) -> np.ndarray:
    """The bar's triangles, each its own colour, flat, as a view of the fixed box sees them."""
    places = solids.mv.point(coordinates(vertices))
    return meshes.draw(places, faces, np.zeros(3), meshes.flat(places, faces), view, background, face_values=colours,
                       colour=meshes.by_face, gloss=solids.gloss)


def side_by_side(bar: core.Mesh, at_rest: core.Point, shapes: list[core.Point], titles: list[str],
                 cell: np.ndarray) -> plt.Figure:
    """The bar in each of the given shapes, side by side, checkered by where each triangle sat at rest."""
    colours = checker(at_rest, bar.faces, cell)
    figure, axes = plt.subplots(1, len(shapes), figsize=(4 * len(shapes), 4), dpi=100, squeeze=False)
    for ax, shape, title in zip(axes[0], shapes, titles):
        show_pixels(ax, picture(shape, bar.faces, colours, fitted(box_view(n_pixels), ax)))
        ax.set_title(title)
    return figure


def animate(bar: core.Mesh, at_rest: core.Point, shapes: Iterable[core.Point], cell: np.ndarray) -> Iterator[np.ndarray]:
    """The bar in each of the given shapes, one frame each, checkered by where each triangle sat at rest."""
    colours = checker(at_rest, bar.faces, cell)
    view = box_view(n_pixels)
    for shape in shapes:
        yield picture(shape, bar.faces, colours, view)
