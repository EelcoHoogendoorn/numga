"""Matched triangle lattices and their field restriction and assembly maps."""

from __future__ import annotations

import numpy as np

from numga.sparse import SparseExtensor, spdiag
from examples import instantiate
from examples.math.isospectral_drums.core import Point, Scalar, ga, mv

Mesh = instantiate("examples.util.mesh", ga).Mesh


# --- plumbing -------------------------------------------------------------------------
def _samples(n_subdivisions: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Integer coordinates and triangulation of one reference tile, and each triangle's right-angled corner, the one
    its square's diagonal faces."""
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
    right_corners = np.concatenate((np.zeros(len(lower_faces), dtype=int), np.ones(len(upper_faces), dtype=int)))
    return column, row, np.concatenate((lower_faces, upper_faces)), right_corners


def triangle(n_subdivisions: int) -> tuple[Point, Mesh]:
    """The reference triangle in the sample order used by restriction maps."""
    column, row, faces, _ = _samples(n_subdivisions)
    vertices = (mv.x.dual() * (column / n_subdivisions)
                + mv.y.dual() * (row / n_subdivisions)).field()           # Point-[samples]
    return vertices, Mesh(len(column), faces)


def drum(tile_vertices: np.ndarray, n_lattice: int, n_subdivisions: int) -> tuple[Point, Mesh, SparseExtensor, SparseExtensor, SparseExtensor, SparseExtensor]:
    """A drum welded from a lattice of triangles, each tile three indices into the row-major square
    lattice: its vertices and mesh, its interior stencil, the maps reading and assembling its tiles, and
    the map that puts its fixed boundary back."""
    corners = np.stack((tile_vertices % n_lattice, tile_vertices // n_lattice), axis=-1)
    n_tiles = len(corners)
    column, row, local_faces, local_right_corners = _samples(n_subdivisions)
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
    # Turning a tile's triangles swaps their last two corners, and with them the right-angled corner's place.
    right_corners = np.where(winding[:, None] > 0, local_right_corners, np.array([0, 2, 1])[local_right_corners]).reshape(-1)

    # The fixed boundary: the vertices of the edges that border a single face; the field lives on the others.
    mesh = Mesh(n_vertices, faces)
    interior = mesh.interior(0)
    read_interior = mesh.reading(interior)                                 # Scalar+[interior <- vertices]
    lift = ~read_interior * Scalar                                         # (Scalar+ <- Scalar+)[vertices <- interior]

    # Each tile's samples read the field at their vertices, zero on the fixed boundary; welding the tiles back
    # averages the samples on each interior vertex, as many as the tiles meeting there.
    restriction = mesh.reading(tile_nodes.reshape(-1)) * ~read_interior    # Scalar+[tile samples <- interior]
    multiplicity = ~restriction * mv.scalar(np.ones((restriction.shape[0], 1))).field()   # Scalar+[interior]
    assembly = spdiag(multiplicity.inverse()) * ~restriction * Scalar      # (Scalar+ <- Scalar+)[interior <- tile samples]
    restriction = restriction * Scalar                                     # (Scalar+ <- Scalar+)[tile samples <- interior]

    # Lattice neighbours are one step along a row or a column: every edge of the triangulation but the squares'
    # diagonals, each triangle's side facing its right-angled corner. They make a graph of their own; between interior
    # sites they couple both ways.
    diagonals = mesh.face_edges[np.arange(len(faces)), right_corners]
    lattice = Mesh(n_vertices, mesh.edges[np.setdiff1d(np.arange(mesh.n_elements[1]), diagonals)])
    tails, heads = (ends * ~read_interior for ends in lattice.incidence[1, 0])   # Scalar+[steps <- interior], Scalar+[steps <- interior]
    neighbors = (~tails * heads + ~heads * tails) * Scalar                 # (Scalar+ <- Scalar+)[interior <- interior]
    return vertices, mesh, neighbors, restriction, assembly, lift
