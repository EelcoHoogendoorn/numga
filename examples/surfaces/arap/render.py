"""GPU flat-shaded rendering of the deformed square bar with a rest-pose checkerboard pattern."""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np

from numga import stack
from examples.surfaces.arap import core
from numga.render import meshes, solids
from examples.util.panels import image_figure

light_colour, dark_colour = np.array([0.98, 0.84, 0.62]), np.array([0.90, 0.45, 0.18])
elevation, azimuth = np.radians([18.0, 26.0])
box_limits = np.array([[-3.3, 3.5], [-1.5, 1.5], [-1.1, 4.0]])
viewport_margin = 0.06
n_pixels = 400
background = np.ones(3)


# --- plumbing -------------------------------------------------------------------------
def checker_colours(topology: core.Topology, rest_vertices: core.Point, tile_size: np.ndarray) -> np.ndarray:
    """Parity of the rest-space grid tile containing each triangle centroid assigns per-face checkerboard colours (`[F, 3]`)."""
    mv = core.mv
    centroids = topology.incidence[2, 0].mean(axis=0) * rest_vertices                # Point-[F]
    tile_planes = stack((mv.x, mv.y, mv.z)) / tile_size                             # [3] Vector+
    parity = np.floor((tile_planes & centroids.batch()[..., None]).to_array()).astype(int).sum(axis=-1) % 2
    return np.where(parity[:, None] == 1, dark_colour, light_colour)


def box_view(width_px: int) -> solids.View:
    """Camera view framing `box_limits` at `width_px` pixels across."""
    corners = solids.mv.point(list(itertools.product(*box_limits)))                 # [8] Point-
    return solids.view([corners], 0.0, elevation, azimuth, viewport_margin, width_px)


def bar_frame(vertices: core.Point, faces: np.ndarray, colours: np.ndarray, view: solids.View) -> np.ndarray:
    """Renders flat-shaded triangles at `vertices` (`Point-[V]`) with per-face `colours` from `view`."""
    places = solids.origin + solids.context.include_from_algebra(vertices.batch())  # [...] Point-
    return meshes.draw(places, faces, np.zeros(3), meshes.flat(places, faces), view, background, face_values=colours,
                       colour=meshes.by_face, gloss=solids.gloss)


def bar_figure(topology: core.Topology, rest_vertices: core.Point, shapes: list[core.Point],
               tile_size: np.ndarray) -> plt.Figure:
    """Renders each vertex field in `shapes` side by side with rest-pose checkerboard colouring."""
    colours = checker_colours(topology, rest_vertices, tile_size)
    view = box_view(n_pixels)
    return image_figure(np.concatenate([bar_frame(shape, topology.elements[2], colours, view) for shape in shapes], axis=1))


def bar_animation(topology: core.Topology, rest_vertices: core.Point, tile_size: np.ndarray,
                  states: Iterable) -> Iterator[np.ndarray]:
    """Yields one RGB frame per state (either `Point-[V]` or `(rigid_vertices, ...)` tuple) with rest-pose checkerboard colouring."""
    colours = checker_colours(topology, rest_vertices, tile_size)
    view = box_view(n_pixels)
    for state in states:
        shape = state[0] if isinstance(state, tuple) else state
        yield bar_frame(shape, topology.elements[2], colours, view)
