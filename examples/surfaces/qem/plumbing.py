"""Load triangle meshes, build edge-collapse neighbourhood connectivity, and greedily select disjoint lowest-cost contractions."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from numga import NumpyContext
from numga.algebras import PGA3D

mv = NumpyContext(PGA3D).multivector


def read_mesh(path: Path, size: float):
    """Load OBJ vertices and triangles, centred and scaled to the requested longest box side."""
    rows = [line.split() for line in path.read_text().splitlines()]
    places = np.array([[float(word) for word in row[1:4]] for row in rows if row[:1] == ["v"]])
    faces = np.array([[int(word.split("/")[0]) - 1 for word in row[1:4]] for row in rows if row[:1] == ["f"]])
    middle = (places.min(axis=0) + places.max(axis=0)) / 2
    places = (places - middle) * (size / np.ptp(places, axis=0).max())
    return mv.point(places), faces


class Topology:
    """Record the edges, vertex stars, and surviving trial triangles of a triangle mesh."""

    def __init__(self, faces: np.ndarray, n_vertices: int):
        self.faces = faces
        self.edges = np.unique(np.sort(np.stack((faces, np.roll(faces, -1, axis=1)), axis=-1).reshape(-1, 2), axis=1), axis=0)
        self.neighbours = [set() for _ in range(n_vertices)]
        self.stars = [set() for _ in range(n_vertices)]
        corners = [[] for _ in range(n_vertices)]
        for face, vertices in enumerate(faces):
            for corner, vertex in enumerate(vertices):
                self.stars[vertex].add(face)
                self.neighbours[vertex].update(vertices)
                corners[vertex].append((face, corner))
        for vertex, neighbours in enumerate(self.neighbours):
            neighbours.discard(vertex)
        self.face_keys = {tuple(sorted(face)) for face in faces}

        trial_edges, trial_faces, trial_corners = [], [], []
        for edge, (first, second) in enumerate(self.edges):
            flanking_faces = self.stars[first] & self.stars[second]
            for vertex in (first, second):
                for face, corner in corners[vertex]:
                    if face not in flanking_faces:
                        trial_edges.append(edge)
                        trial_faces.append(face)
                        trial_corners.append(corner)
        self.trial_edges = np.asarray(trial_edges, dtype=int)
        self.trial_faces = np.asarray(trial_faces, dtype=int)
        self.trial_corners = np.asarray(trial_corners, dtype=int)


def select_edges(topology: Topology, errors, valid_trials: np.ndarray, max_edges: int) -> np.ndarray:
    """Select the lowest-cost legal contractions with disjoint changed triangle neighbourhoods."""
    costs = errors.to_array()
    valid_edges = np.isfinite(costs)
    np.logical_and.at(valid_edges, topology.trial_edges, valid_trials)
    locked_vertices = np.zeros(len(topology.neighbours), dtype=bool)
    chosen_edges = []
    for edge in np.argsort(np.where(valid_edges, costs, np.inf)):
        if not valid_edges[edge]:
            break
        first, second = topology.edges[edge]
        if locked_vertices[first] or locked_vertices[second]:
            continue
        shared_faces = topology.stars[first] & topology.stars[second]
        if len(shared_faces) != 2:
            continue
        common_neighbours = topology.neighbours[first] & topology.neighbours[second]
        opposite_vertices = set(topology.faces[list(shared_faces)].reshape(-1)) - {first, second}
        if common_neighbours != opposite_vertices:
            continue
        # The vertex-link test also excludes identifying two surviving triangles.
        # In a tetrahedral neighbourhood their opposite edge belongs to both links.
        if tuple(sorted((first, *common_neighbours))) in topology.face_keys and tuple(sorted((second, *common_neighbours))) in topology.face_keys:
            continue
        chosen_edges.append(edge)
        locked_vertices[list(topology.neighbours[first] | topology.neighbours[second] | {first, second})] = True
        if len(chosen_edges) == max_edges:
            break
    return np.asarray(chosen_edges, dtype=int)


def contract(topology: Topology, chosen_edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Identify chosen endpoint pairs, discard their collapsed faces, and compact vertex indices."""
    mapping = np.arange(len(topology.neighbours))
    first, second = topology.edges[chosen_edges].T
    mapping[second] = first
    faces = mapping[topology.faces]
    surviving_faces = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 2] != faces[:, 0])
    kept_vertices, reindexed_faces = np.unique(faces[surviving_faces], return_inverse=True)
    return kept_vertices, reindexed_faces.reshape(-1, 3)
