"""The faces each edge's collapse would move are found, and disjoint lowest-cost
contractions are selected and applied, each giving the coarser mesh and the maps placing its vertices."""

from __future__ import annotations

import numpy as np

from numga.algebras import PGA3D
from numga.sparse import SparseExtensor
from examples import instantiate

mesh_module = instantiate("examples.util.mesh", PGA3D)
Mesh, as_scalar, ctx = mesh_module.Mesh, mesh_module.as_scalar, mesh_module.ctx
Scalar = PGA3D.gatype.scalar()                      # Scalar+


def trials(mesh: Mesh) -> tuple[SparseExtensor, SparseExtensor]:
    """The faces an edge's collapse moves: each face with one of the edge's ends as a corner, but not beside the
    edge, a trial. As the map reading each trial's edge, `Scalar+[trials <- edges]`, and the map reading its face at
    the corner on the edge's end, a batch over the corners, `[corners] Scalar+[trials <- faces]`, one at that corner
    and zero at the others."""
    # Each edge coupled to the faces at its ends: at each corner, how many of the edge's ends that corner is.
    touching = mesh.incidence[1, 0].sum(axis=0) * ~mesh.incidence[2, 0]       # [corners] Scalar+[edges <- faces]
    # The couplings read out of the map, one trial each where a single corner is an end: a face with two is beside it.
    shares = touching.cells.to_array()                                         # [corners, couplings]
    alone = shares.sum(axis=0) == 1                                            # [couplings]
    n_trials = np.count_nonzero(alone)
    at_edges = SparseExtensor.selection(ctx, touching.rows[alone], len(mesh.edges))
    at_corners = SparseExtensor.from_indices(as_scalar(shares[:, alone]), np.arange(n_trials), touching.columns[alone],
                                             (n_trials, len(mesh.faces)))
    return at_edges, at_corners


def select_edges(mesh: Mesh, errors: Scalar, legal: np.ndarray, max_edges: int) -> np.ndarray:
    """The lowest-cost legal contractions whose neighbourhoods are disjoint, at most `max_edges` of them; a
    contraction keeps the closed surface a manifold where the edge's ends share no neighbours but the two across
    the edge."""
    # Each vertex with every vertex an edge joins to it, itself included: the edges' ends read back onto their ends.
    reach = ~mesh.incidence[1, 0].sum(axis=0) * mesh.incidence[1, 0].sum(axis=0)   # Scalar+[vertices <- vertices]
    neighbourhoods = np.split(reach.columns, np.flatnonzero(np.diff(reach.rows)) + 1)
    costs = errors.to_array()
    locked_vertices = np.zeros(mesh.n_vertices, dtype=bool)
    chosen_edges = []
    for edge in np.argsort(np.where(legal, costs, np.inf)):
        if not legal[edge]:
            break
        first, second = mesh.edges[edge]
        if locked_vertices[first] or locked_vertices[second]:
            continue
        # The two neighbourhoods share the ends themselves and the two vertices across the edge, and no more.
        if len(np.intersect1d(neighbourhoods[first], neighbourhoods[second])) != 4:
            continue
        chosen_edges.append(edge)
        locked_vertices[neighbourhoods[first]] = locked_vertices[neighbourhoods[second]] = True
        if len(chosen_edges) == max_edges:
            break
    return np.asarray(chosen_edges, dtype=int)


def contract(mesh: Mesh, chosen_edges: np.ndarray) -> tuple[Mesh, SparseExtensor, SparseExtensor]:
    """The coarser mesh with the ends of each chosen edge identified and the faces beside it gone, its vertices
    renumbered in order; and the maps placing its vertices, from the vertices the contractions leave in place,
    `Scalar+[coarser <- vertices]`, and from the chosen edges, at the vertex each merges into,
    `Scalar+[coarser <- edges]`."""
    mapping = np.arange(mesh.n_vertices)
    first, second = mesh.edges[chosen_edges].T
    mapping[second] = first
    faces = mapping[mesh.faces]
    surviving_faces = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 2] != faces[:, 0])
    kept_vertices, reindexed_faces = np.unique(faces[surviving_faces], return_inverse=True)
    n_coarser = len(kept_vertices)
    in_place = np.flatnonzero(~np.isin(kept_vertices, first))
    kept = SparseExtensor.from_indices(as_scalar(np.ones(len(in_place))), in_place, kept_vertices[in_place],
                                       (n_coarser, mesh.n_vertices))
    merged = SparseExtensor.from_indices(as_scalar(np.ones(len(chosen_edges))), np.searchsorted(kept_vertices, first),
                                         chosen_edges, (n_coarser, len(mesh.edges)))
    return Mesh(n_coarser, reindexed_faces.reshape(-1, 3)), kept, merged
