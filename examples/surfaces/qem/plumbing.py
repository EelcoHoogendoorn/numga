"""Mesh input, connectivity and scalar priority selection for the QEM contractions."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def read_mesh(path: Path, size: float):
    """OBJ vertices and triangles, centred and scaled to the requested longest box side."""
    from examples.surfaces.qem.core import mv

    rows = [line.split() for line in path.read_text().splitlines()]
    places = np.array([[float(word) for word in row[1:4]] for row in rows if row[:1] == ["v"]])
    faces = np.array([[int(word.split("/")[0]) - 1 for word in row[1:4]] for row in rows if row[:1] == ["f"]])
    middle = (places.min(axis=0) + places.max(axis=0)) / 2
    places = (places - middle) * (size / np.ptp(places, axis=0).max())
    return mv.point(places), faces


class Topology:
    """The edges, vertex stars and surviving trial triangles of one triangle mesh."""

    def __init__(self, faces: np.ndarray, vertex_count: int):
        self.faces = faces
        self.edges = np.unique(np.sort(np.stack((faces, np.roll(faces, -1, axis=1)), axis=-1).reshape(-1, 2), axis=1), axis=0)
        self.neighbours = [set() for _ in range(vertex_count)]
        self.stars = [set() for _ in range(vertex_count)]
        corners = [[] for _ in range(vertex_count)]
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
            removed = self.stars[first] & self.stars[second]
            for vertex in (first, second):
                for face, corner in corners[vertex]:
                    if face not in removed:
                        trial_edges.append(edge)
                        trial_faces.append(face)
                        trial_corners.append(corner)
        self.trial_edges = np.asarray(trial_edges, dtype=int)
        self.trial_faces = np.asarray(trial_faces, dtype=int)
        self.trial_corners = np.asarray(trial_corners, dtype=int)


def select_edges(topology: Topology, errors, consistent: np.ndarray, limit: int) -> np.ndarray:
    """Lowest-cost legal contractions with disjoint changed triangle neighbourhoods."""
    costs = errors.to_array()
    admissible = np.isfinite(costs)
    np.logical_and.at(admissible, topology.trial_edges, consistent)
    blocked = np.zeros(len(topology.neighbours), dtype=bool)
    selected = []
    for edge in np.argsort(np.where(admissible, costs, np.inf)):
        if not admissible[edge]:
            break
        first, second = topology.edges[edge]
        if blocked[first] or blocked[second]:
            continue
        shared = topology.stars[first] & topology.stars[second]
        if len(shared) != 2:
            continue
        common = topology.neighbours[first] & topology.neighbours[second]
        opposite = set(topology.faces[list(shared)].reshape(-1)) - {first, second}
        if common != opposite:
            continue
        # The vertex-link test also excludes identifying two surviving triangles.
        # In a tetrahedral neighbourhood their opposite edge belongs to both links.
        if tuple(sorted((first, *common))) in topology.face_keys and tuple(sorted((second, *common))) in topology.face_keys:
            continue
        selected.append(edge)
        blocked[list(topology.neighbours[first] | topology.neighbours[second] | {first, second})] = True
        if len(selected) == limit:
            break
    return np.asarray(selected, dtype=int)


def contract(topology: Topology, selected: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Identify selected endpoint pairs, discard their collapsed faces and compact indices."""
    mapping = np.arange(len(topology.neighbours))
    first, second = topology.edges[selected].T
    mapping[second] = first
    faces = mapping[topology.faces]
    survives = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 2] != faces[:, 0])
    retained, reindexed = np.unique(faces[survives], return_inverse=True)
    return retained, reindexed.reshape(-1, 3)
