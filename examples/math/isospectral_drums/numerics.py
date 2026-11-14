"""Subdivided right-triangle lattices build welded drum topologies alongside their interior stencils, tile restrictions, and seam assemblies."""

from __future__ import annotations

import numpy as np

from numga.sparse import SparseExtensor, spdiag
from numga.topology import Topology
from examples.math.isospectral_drums.core import Point, Scalar, mv


# --- plumbing -------------------------------------------------------------------------
def _samples(n_subdivisions: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Lattice leg coordinates, local triangle faces, and right-angle corner indices subdivide one reference right triangle into `n_subdivisions` steps per leg."""
    row, column = np.indices((n_subdivisions + 1, n_subdivisions + 1))
    triangle_mask = row + column <= n_subdivisions
    column, row = column[triangle_mask], row[triangle_mask]
    n_samples = len(column)
    # Split each lattice square along the diagonal parallel to the reference hypotenuse.
    local_nodes = np.full((n_subdivisions + 1, n_subdivisions + 1), -1)
    local_nodes[row, column] = np.arange(n_samples)
    lower_mask = row + column < n_subdivisions
    upper_mask = row + column < n_subdivisions - 1
    lower_faces = np.stack((local_nodes[row[lower_mask], column[lower_mask]],
                            local_nodes[row[lower_mask], column[lower_mask] + 1],
                            local_nodes[row[lower_mask] + 1, column[lower_mask]]), axis=-1)
    upper_faces = np.stack((local_nodes[row[upper_mask], column[upper_mask] + 1],
                            local_nodes[row[upper_mask] + 1, column[upper_mask] + 1],
                            local_nodes[row[upper_mask] + 1, column[upper_mask]]), axis=-1)
    right_corners = np.concatenate((np.zeros(len(lower_faces), dtype=int), np.ones(len(upper_faces), dtype=int)))
    return column, row, np.concatenate((lower_faces, upper_faces)), right_corners


def triangle(n_subdivisions: int) -> tuple[Point, Topology]:
    """Subdivided reference right triangle pairs sample vertices (`Point-[V]`) in restriction order with their triangulation."""
    column, row, faces = _samples(n_subdivisions)[:3]
    vertices = (mv.x.dual() * (column / n_subdivisions)
                + mv.y.dual() * (row / n_subdivisions)).field()         # Point-[V]
    return vertices, Topology(len(column), faces)


def drum(
    tile_vertices: np.ndarray, n_lattice: int, n_subdivisions: int,
) -> tuple[Point, Topology, SparseExtensor, SparseExtensor, SparseExtensor, SparseExtensor]:
    """Welded triangles on the `n_lattice` by `n_lattice` grid define a drum's vertices (`Point-[V]`), topology, interior neighbour sum, tile restriction, seam assembly, and zero-boundary extension."""
    corners = np.stack((tile_vertices % n_lattice, tile_vertices // n_lattice), axis=-1)
    n_tiles = len(corners)
    column, row, local_faces, local_right_corners = _samples(n_subdivisions)
    n_samples = len(column)

    # Integer coordinates identify coincident samples across tile seams.
    locations = (corners[:, 0, None] * n_subdivisions
                 + (corners[:, 1] - corners[:, 0])[:, None] * column[None, :, None]
                 + (corners[:, 2] - corners[:, 0])[:, None] * row[None, :, None])
    nodes, tile_nodes = np.unique(locations.reshape(-1, 2), axis=0, return_inverse=True)
    tile_nodes = tile_nodes.reshape(n_tiles, n_samples)
    vertices = (mv.x.dual() * (nodes[:, 0] / n_subdivisions)
                + mv.y.dual() * (nodes[:, 1] / n_subdivisions)).field()   # Point-[V]

    faces = tile_nodes[:, local_faces]
    frame = corners[:, 1:] - corners[:, :1]
    winding = frame[:, 0, 0] * frame[:, 1, 1] - frame[:, 0, 1] * frame[:, 1, 0]
    faces = np.where(winding[:, None, None] > 0, faces, faces[..., [0, 2, 1]]).reshape(-1, 3)
    # Reversing triangle winding swaps the last two corners and moves the right-angle corner index with them.
    right_corners = np.where(winding[:, None] > 0, local_right_corners, np.array([0, 2, 1])[local_right_corners]).reshape(-1)

    topology = Topology(len(nodes), faces)
    V, E, F = topology.n_elements
    interior = topology.interior[0]
    interior_reading = topology.reading(interior, 0)                    # Scalar+[V_interior <- V]
    extension = interior_reading.adjugate() * Scalar                    # (Scalar+ <- Scalar+)[V <- V_interior]

    # Restriction reads interior values onto each tile's samples; seam assembly averages coincident tile samples.
    restriction = topology.reading(tile_nodes.reshape(-1), 0) * interior_reading.adjugate()  # Scalar+[n_tiles * n_samples <- V_interior]
    multiplicity = restriction.adjugate() * mv.scalar(np.ones((n_tiles * n_samples, 1))).field()  # Scalar+[V_interior]
    assembly = spdiag(multiplicity.inverse()) * restriction.adjugate() * Scalar  # (Scalar+ <- Scalar+)[V_interior <- n_tiles * n_samples]
    restriction = restriction * Scalar                                  # (Scalar+ <- Scalar+)[n_tiles * n_samples <- V_interior]

    # Lattice neighbours step along grid rows and columns, omitting the hypotenuse edge opposite each right-angle corner.
    diagonals = topology.boundary_indices[1][np.arange(F), right_corners]
    lattice = topology.subset(np.setdiff1d(np.arange(E), diagonals), 1)
    tails, heads = (ends * interior_reading.adjugate() for ends in lattice.incidence[1, 0])  # Scalar+[E_lattice <- V_interior], Scalar+[E_lattice <- V_interior]
    neighbors = (tails.adjugate() * heads + heads.adjugate() * tails) * Scalar  # (Scalar+ <- Scalar+)[V_interior <- V_interior]
    return vertices, topology, neighbors, restriction, assembly, extension
