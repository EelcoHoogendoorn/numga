"""Matched triangle lattices and their field restriction and assembly maps."""

from __future__ import annotations

import numpy as np

from numga.sparse import SparseExtensor
from examples.math.isospectral_drums.core import Scalar, Vector, mv


def _samples(n_subdivisions: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integer coordinates and triangulation of one reference tile."""
    row, column = np.indices((n_subdivisions + 1, n_subdivisions + 1))
    within = row + column <= n_subdivisions
    column, row = column[within], row[within]
    n_samples = len(column)
    # Split each lattice square along the diagonal shared by the reference triangle.
    local_nodes = np.full((n_subdivisions + 1, n_subdivisions + 1), -1)
    local_nodes[row, column] = np.arange(n_samples)
    lower = row + column < n_subdivisions
    upper = row + column < n_subdivisions - 1
    lower_faces = np.stack((local_nodes[row[lower], column[lower]],
                            local_nodes[row[lower], column[lower] + 1],
                            local_nodes[row[lower] + 1, column[lower]]), axis=-1)
    upper_faces = np.stack((local_nodes[row[upper], column[upper] + 1],
                            local_nodes[row[upper] + 1, column[upper] + 1],
                            local_nodes[row[upper] + 1, column[upper]]), axis=-1)
    return column, row, np.concatenate((lower_faces, upper_faces))


def triangle(n_subdivisions: int) -> tuple[Vector, np.ndarray]:
    """The reference triangle in the sample order used by restriction maps."""
    column, row, faces = _samples(n_subdivisions)
    vertices = (mv.x * (column / n_subdivisions)
                + mv.y * (row / n_subdivisions)).field()                  # Vector[samples]
    return vertices, faces


def drum(tile_vertices: np.ndarray, n_lattice: int, n_subdivisions: int) -> tuple[Vector, np.ndarray, SparseExtensor, SparseExtensor, SparseExtensor, SparseExtensor]:
    """A drum welded from a lattice of triangles, each tile three indices into the row-major square
    lattice: its vertices and faces, its interior stencil, the maps reading and assembling its tiles, and
    the map that puts its fixed boundary back."""
    corners = np.stack((tile_vertices % n_lattice, tile_vertices // n_lattice), axis=-1)
    n_tiles = len(corners)
    column, row, local_faces = _samples(n_subdivisions)
    n_samples = len(column)

    # Integer coordinates identify coincident samples exactly, including tile seams.
    locations = (corners[:, 0, None] * n_subdivisions
                 + (corners[:, 1] - corners[:, 0])[:, None] * column[None, :, None]
                 + (corners[:, 2] - corners[:, 0])[:, None] * row[None, :, None])
    nodes, tile_nodes = np.unique(locations.reshape(-1, 2), axis=0, return_inverse=True)
    tile_nodes = tile_nodes.reshape(n_tiles, n_samples)
    n_vertices = len(nodes)
    vertices = (mv.x * (nodes[:, 0] / n_subdivisions)
                + mv.y * (nodes[:, 1] / n_subdivisions)).field()            # Vector[vertices]

    faces = tile_nodes[:, local_faces]
    frame = corners[:, 1:] - corners[:, :1]
    winding = frame[:, 0, 0] * frame[:, 1, 1] - frame[:, 0, 1] * frame[:, 1, 0]
    faces = np.where(winding[:, None, None] > 0, faces, faces[..., [0, 2, 1]]).reshape(-1, 3)

    # Only edges belonging to one large triangle are fixed drum boundaries.
    corner_nodes, corner_ids = np.unique(corners.reshape(-1, 2), axis=0, return_inverse=True)
    corner_ids = corner_ids.reshape(n_tiles, 3)
    edges = np.concatenate((corner_ids[:, [0, 1]], corner_ids[:, [1, 2]], corner_ids[:, [2, 0]]))
    edges, counts = np.unique(np.sort(edges, axis=-1), axis=0, return_counts=True)
    boundary_edges = corner_nodes[edges[counts == 1]] * n_subdivisions
    edge_direction = boundary_edges[:, 1] - boundary_edges[:, 0]
    relative = nodes[:, None] - boundary_edges[None, :, 0]
    cross = relative[..., 0] * edge_direction[:, 1] - relative[..., 1] * edge_direction[:, 0]
    along = np.sum(relative * edge_direction, axis=-1)
    length_squared = np.sum(edge_direction ** 2, axis=-1)
    boundary = np.any((cross == 0) & (along >= 0) & (along <= length_squared), axis=-1)
    interior = np.flatnonzero(~boundary)
    n_interior = len(interior)
    interior_ids = np.full(n_vertices, -1)
    interior_ids[interior] = np.arange(n_interior)

    tile_interior = interior_ids[tile_nodes].reshape(-1)
    tile_rows = np.flatnonzero(tile_interior >= 0)
    tile_columns = tile_interior[tile_rows]
    unit_cells = mv.scalar(np.ones((len(tile_rows), 1))) * Scalar           # [couplings] Scalar <- Scalar
    restriction = SparseExtensor.from_indices(unit_cells, tile_rows, tile_columns,
                                              (n_tiles * n_samples, n_interior))
    multiplicity = np.bincount(tile_columns, minlength=n_interior)
    assembly = SparseExtensor.from_indices(unit_cells / multiplicity[tile_columns],
                                            tile_columns, tile_rows, (n_interior, n_tiles * n_samples))
    lift = SparseExtensor.from_indices(mv.scalar(np.ones((n_interior, 1))) * Scalar,
                                        interior, np.arange(n_interior), (n_vertices, n_interior))

    # A one-cell border makes every nearest-neighbour lookup an ordinary index.
    offset = 1 - nodes.min(axis=0)
    lookup = np.full(tuple(nodes.max(axis=0) + offset + 2), -1)
    shifted = nodes[interior] + offset
    lookup[shifted[:, 0], shifted[:, 1]] = np.arange(n_interior)
    steps = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    adjacent_nodes = shifted[:, None] + steps
    adjacent = lookup[adjacent_nodes[..., 0], adjacent_nodes[..., 1]]
    tails, directions = np.nonzero(adjacent >= 0)
    neighbors = SparseExtensor.from_indices(mv.scalar(np.ones((len(tails), 1))) * Scalar,
                                             tails, adjacent[tails, directions], (n_interior, n_interior))
    return vertices, faces, neighbors, restriction, assembly, lift
