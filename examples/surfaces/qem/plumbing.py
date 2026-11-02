"""The faces each edge's collapse would move are found, and disjoint lowest-cost
contractions are selected and applied, each giving the coarser mesh and the maps placing its vertices."""

from __future__ import annotations

import numpy as np

from numga.algebras import PGA3D
from numga.sparse import SparseExtensor
from examples.util.topology import Topology

Scalar = PGA3D.gatype.scalar()                      # Scalar+


def trials(topology: Topology) -> tuple[SparseExtensor, SparseExtensor]:
    """The faces an edge's collapse moves: each face with one of the edge's ends as a corner, but not beside the
    edge, a trial. As the map reading each trial's edge, `Scalar+[trials <- edges]`, and the map reading its face at
    the corner on the edge's end, a batch over the corners, `[corners] Scalar+[trials <- faces]`, one at that corner
    and zero at the others."""
    # Each edge coupled to the faces at its ends: at each corner, how many of the edge's ends that corner is.
    touching = topology.incidence[1, 0].sum(axis=0) * ~topology.incidence[2, 0]  # [corners] Scalar+[edges <- faces]
    # The couplings read out of the map, one trial each where a single corner is an end: a face with two is beside it.
    shares = touching.cells                                                    # [corners, couplings]
    alone = shares.sum(axis=0) == 1                                            # [couplings]
    n_trials = np.count_nonzero(alone)
    at_edges = topology.reading(touching.rows[alone], 1)
    at_corners = SparseExtensor.from_indices(shares[:, alone], np.arange(n_trials), touching.columns[alone],
                                             (n_trials, topology.n_elements[2]))
    return at_edges, at_corners


def select_edges(topology: Topology, errors: Scalar, legal: np.ndarray, max_edges: int) -> np.ndarray:
    """The lowest-cost legal contractions whose neighbourhoods are disjoint, at most `max_edges` of them; a
    contraction keeps the closed surface a manifold where the edge's ends share no neighbours but the two across
    the edge."""
    # Each vertex with every vertex an edge joins to it, itself included: the edges' ends read back onto their ends.
    reach = ~topology.incidence[1, 0].sum(axis=0) * topology.incidence[1, 0].sum(axis=0)  # Scalar+[vertices <- vertices]
    neighbourhoods = np.split(reach.columns, np.flatnonzero(np.diff(reach.rows)) + 1)
    costs = errors.to_array()
    locked_vertices = np.zeros(topology.n_vertices, dtype=bool)
    chosen_edges = []
    for edge in np.argsort(np.where(legal, costs, np.inf)):
        if not legal[edge]:
            break
        first, second = topology.elements[1][edge]
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


def contract(topology: Topology, chosen_edges: np.ndarray) -> tuple[Topology, SparseExtensor, SparseExtensor]:
    """The coarser mesh with the ends of each chosen edge identified and the faces beside it gone, its vertices
    renumbered in order; and the maps placing its vertices, from the vertices the contractions leave in place,
    `Scalar+[coarser <- vertices]`, and from the chosen edges, at the vertex each merges into,
    `Scalar+[coarser <- edges]`."""
    mapping = np.arange(topology.n_vertices)
    first, second = topology.elements[1][chosen_edges].T
    mapping[second] = first
    faces = mapping[topology.elements[2]]
    surviving_faces = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 2] != faces[:, 0])
    kept_vertices, reindexed_faces = np.unique(faces[surviving_faces], return_inverse=True)
    n_coarser = len(kept_vertices)
    in_place = np.flatnonzero(~np.isin(kept_vertices, first))
    kept = SparseExtensor.from_indices(np.ones(len(in_place)), in_place, kept_vertices[in_place],
                                       (n_coarser, topology.n_vertices))
    merged = SparseExtensor.from_indices(np.ones(len(chosen_edges)), np.searchsorted(kept_vertices, first),
                                         chosen_edges, (n_coarser, topology.n_elements[1]))
    return Topology(n_coarser, reindexed_faces.reshape(-1, 3)), kept, merged
