"""Matched triangle lattices and their field restriction and assembly maps."""

from __future__ import annotations

import numpy as np

from numga.sparse import SparseExtensor
from examples import instantiate
from examples.math.isospectral_drums.core import Point, Scalar, ga, mv

Mesh = instantiate("examples.util.mesh", ga).Mesh


# --- plumbing -------------------------------------------------------------------------
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


def triangle(n_subdivisions: int) -> tuple[Point, np.ndarray]:
    """The reference triangle in the sample order used by restriction maps."""
    column, row, faces = _samples(n_subdivisions)
    vertices = (mv.x.dual() * (column / n_subdivisions)
                + mv.y.dual() * (row / n_subdivisions)).field()           # Point-[samples]
    return vertices, faces


def drum(tile_vertices: np.ndarray, n_lattice: int, n_subdivisions: int) -> tuple[Point, np.ndarray, SparseExtensor, SparseExtensor, SparseExtensor, SparseExtensor]:
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
    vertices = (mv.x.dual() * (nodes[:, 0] / n_subdivisions)
                + mv.y.dual() * (nodes[:, 1] / n_subdivisions)).field()     # Point-[vertices]

    faces = tile_nodes[:, local_faces]
    frame = corners[:, 1:] - corners[:, :1]
    winding = frame[:, 0, 0] * frame[:, 1, 1] - frame[:, 0, 1] * frame[:, 1, 0]
    faces = np.where(winding[:, None, None] > 0, faces, faces[..., [0, 2, 1]]).reshape(-1, 3)

    # The fixed boundary: the vertices of the edges that border a single face.
    mesh = Mesh(n_vertices, faces)
    boundary = np.isin(np.arange(n_vertices), mesh.edges[mesh.boundary_edges])
    interior = np.flatnonzero(~boundary)
    n_interior = len(interior)
    interior_ids = np.full(n_vertices, -1)
    interior_ids[interior] = np.arange(n_interior)

    tile_interior = interior_ids[tile_nodes].reshape(-1)
    tile_rows = np.flatnonzero(tile_interior >= 0)
    tile_columns = tile_interior[tile_rows]
    unit_cells = mv.scalar(np.ones((len(tile_rows), 1))) * Scalar           # [couplings] Scalar+ <- Scalar+
    restriction = SparseExtensor.from_indices(unit_cells, tile_rows, tile_columns,
                                              (n_tiles * n_samples, n_interior))
    multiplicity = np.bincount(tile_columns, minlength=n_interior)
    assembly = SparseExtensor.from_indices(unit_cells / multiplicity[tile_columns],
                                            tile_columns, tile_rows, (n_interior, n_tiles * n_samples))
    lift = SparseExtensor.from_indices(mv.scalar(np.ones((n_interior, 1))) * Scalar,
                                        interior, np.arange(n_interior), (n_vertices, n_interior))

    # Lattice neighbours are the edges of the triangulation one step along a row or a column, not across a diagonal,
    # between interior sites, coupled both ways.
    steps = np.abs(nodes[mesh.edges[:, 1]] - nodes[mesh.edges[:, 0]]).sum(axis=-1)
    ends = interior_ids[mesh.edges[(steps == 1) & np.all(interior_ids[mesh.edges] >= 0, axis=-1)]]
    tails, heads = np.concatenate((ends, ends[:, ::-1])).T
    neighbors = SparseExtensor.from_indices(mv.scalar(np.ones((len(tails), 1))) * Scalar,
                                             tails, heads, (n_interior, n_interior))
    return vertices, faces, neighbors, restriction, assembly, lift
