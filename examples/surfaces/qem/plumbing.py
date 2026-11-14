"""Combinatorial trial enumeration, disjoint edge selection, and batch edge contraction for triangulated surfaces."""

from __future__ import annotations

import numpy as np

from numga.algebras import PGA3D
from numga.sparse import Sparse
from numga.topology import Topology

Scalar = PGA3D.gatype.scalar()                      # Scalar+


def trials(topology: Topology) -> tuple[Sparse, Sparse]:
    """Triangles sharing a single corner with an edge move without degenerating when that edge collapses, paired via `Scalar+[n_trials <- E]` and `[3] Scalar+[n_trials <- F]`."""
    V, E, F = topology.n_elements
    edges, faces = topology.elements[1], topology.elements[2]                   # [E, 2], [F, 3]
    touching = topology.incidence[1, 0].sum(axis=0) * topology.incidence[2, 0].adjugate()   # Scalar+[E <- F]
    # A coupled `(edge, face)` pair moves without degenerating when the triangle contains one endpoint of the edge rather than both.
    trial_edges, trial_faces = touching.rows, touching.columns
    on_end = (faces[trial_faces, :, None] == edges[trial_edges, None, :]).any(axis=-1)      # [n_couplings, 3]
    alone = on_end.sum(axis=1) == 1                                             # [n_couplings]
    n_trials = int(np.count_nonzero(alone))
    at_edges = topology.reading(trial_edges[alone], 1)                          # Scalar+[n_trials <- E]
    at_corners = Sparse.from_indices(on_end[alone].T, np.arange(n_trials), trial_faces[alone], (n_trials, F))
    return at_edges, at_corners


def select_edges(topology: Topology, errors: Scalar, legal: np.ndarray, n_max_edges: int) -> np.ndarray:
    """Greedy selection picks up to `n_max_edges` lowest-cost legal edges with disjoint vertex stars satisfying the link condition (`4` shared vertices across endpoints and itself)."""
    V, E, F = topology.n_elements
    # Compose edge-to-vertex incidence with its adjugate to gather each vertex's closed one-ring neighbourhood.
    reach = topology.incidence[1, 0].sum(axis=0).adjugate() * topology.incidence[1, 0].sum(axis=0)  # Scalar+[V <- V]
    neighbourhoods = np.split(reach.columns, np.flatnonzero(np.diff(reach.rows)) + 1)
    costs = errors.to_array()                                                   # [E]
    locked_vertices = np.zeros(V, dtype=bool)
    chosen_edges = []
    for edge in np.argsort(np.where(legal, costs, np.inf)):
        if not legal[edge]:
            break
        first, second = topology.elements[1][edge]
        if locked_vertices[first] or locked_vertices[second]:
            continue
        # On a closed manifold, a collapsible edge's closed endpoint stars share only the two endpoints and the two opposite corners.
        if len(np.intersect1d(neighbourhoods[first], neighbourhoods[second])) != 4:
            continue
        chosen_edges.append(edge)
        locked_vertices[neighbourhoods[first]] = locked_vertices[neighbourhoods[second]] = True
        if len(chosen_edges) == n_max_edges:
            break
    return np.asarray(chosen_edges, dtype=int)


def contract(topology: Topology, chosen_edges: np.ndarray) -> tuple[Topology, Sparse, Sparse]:
    """Identifying the endpoints of `chosen_edges` drops their flanking triangles and returns the compacted `Topology` with vertex gather maps `Scalar+[V_coarser <- V]` and `Scalar+[V_coarser <- E]`."""
    V, E, F = topology.n_elements
    mapping = np.arange(V)                                                      # [V]
    first, second = topology.elements[1][chosen_edges].T
    mapping[second] = first
    faces = mapping[topology.elements[2]]                                       # [F, 3]
    surviving_faces = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 2] != faces[:, 0])
    kept_vertices, reindexed_faces = np.unique(faces[surviving_faces], return_inverse=True)
    V_coarser = len(kept_vertices)
    in_place = np.flatnonzero(~np.isin(kept_vertices, first))
    kept = Sparse.from_indices(np.ones(len(in_place)), in_place, kept_vertices[in_place], (V_coarser, V))
    merged = Sparse.from_indices(np.ones(len(chosen_edges)), np.searchsorted(kept_vertices, first), chosen_edges, (V_coarser, E))
    return Topology(V_coarser, reindexed_faces.reshape(-1, 3)), kept, merged
