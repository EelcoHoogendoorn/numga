"""Triangulated surface loading and low-discrepancy volume quadrature for solid deformation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.stats import qmc

from numga import NumpyContext, stack
from numga.algebras import VGA3D
from examples.util import shapes
from numga.topology import Topology

mv = NumpyContext(VGA3D).multivector
# Points are antivectors.
Antivector = VGA3D.gatype.antivector()
n_ray_batch = 256


def coordinates(points: Antivector) -> np.ndarray:
    """Pairing `points` with the three coordinate planes reads Cartesian positions (`[..., 3]`)."""
    return (stack((mv.x, mv.y, mv.z)) & points[..., None]).to_array()


def interior_mask(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Ray-crossing parity classifies `points` (`[n_points, 3]`) against a closed triangulated surface (`[3, F, 3]`)."""
    origin = triangles[0]
    first = triangles[1] - origin
    second = triangles[2] - origin
    determinant = first[:, 1] * second[:, 2] - first[:, 2] * second[:, 1]
    transverse = determinant != 0
    origin, first, second = origin[transverse], first[transverse], second[transverse]
    determinant = determinant[transverse]
    inside = np.zeros(len(points), dtype=bool)
    for start in range(0, len(points), n_ray_batch):
        samples = points[start:start + n_ray_batch]
        offset_y = samples[:, None, 1] - origin[None, :, 1]
        offset_z = samples[:, None, 2] - origin[None, :, 2]
        first_weight = (offset_y * second[:, 2] - offset_z * second[:, 1]) / determinant
        second_weight = (first[:, 1] * offset_z - first[:, 2] * offset_y) / determinant
        crosses = (first_weight >= 0) & (second_weight >= 0) & (first_weight + second_weight <= 1)
        crossing_x = origin[:, 0] + first_weight * first[:, 0] + second_weight * second[:, 0]
        inside[start:start + len(samples)] = (crosses & (crossing_x > samples[:, None, 0])).sum(axis=-1) % 2 == 1
    return inside


def volume_samples(topology: Topology, vertices: Antivector, n_samples: int) -> tuple[Antivector, np.ndarray]:
    """Low-discrepancy Sobol sampling inside the closed surface `topology` draws `n_samples` equal-volume quadrature points (`[n_samples] Antivector-`)."""
    V, E, F = topology.n_elements
    corners = topology.incidence[2, 0] * vertices                # [3] Antivector-[F]
    # The enclosed volume equals one sixth of the summed three-corner joins from the origin.
    volume = float((corners[0] & corners[1] & corners[2]).sites.sum().abs().to_array()) / 6
    places = coordinates(vertices.batch())                       # [V, 3]
    lower, upper = places.min(axis=0), places.max(axis=0)
    triangles = coordinates(corners.batch())                     # [3, F, 3]
    occupancy = volume / np.prod(upper - lower)
    n_candidates = 2 ** int(np.ceil(np.log2(1.2 * n_samples / occupancy)))
    sequence = qmc.Sobol(3, scramble=True, seed=0)
    accepted_batches = []
    n_inside = 0
    while n_inside < n_samples:
        candidates = lower + sequence.random(n_candidates) * (upper - lower)
        accepted = candidates[interior_mask(candidates, triangles)]
        accepted_batches.append(accepted)
        n_inside += len(accepted)
    positions = np.concatenate(accepted_batches)[:n_samples]
    return mv.vector(positions).dual(), np.full(n_samples, volume / n_samples)


def center_indices(points: np.ndarray, n_centers: int) -> np.ndarray:
    """Farthest-point sampling over `points` selects `n_centers` indices covering the solid and its extremities (`[n_centers]`)."""
    indices = np.empty(n_centers, dtype=int)
    indices[0] = np.argmin(np.sum((points - points.mean(axis=0)) ** 2, axis=-1))
    distances = np.full(len(points), np.inf)
    for index in range(n_centers - 1):
        distances = np.minimum(distances, np.sum((points - points[indices[index]]) ** 2, axis=-1))
        indices[index + 1] = np.argmax(distances)
    return indices


def load_solid(path: Path, height: float, n_quadrature: int,
               n_centers: int) -> tuple[Topology, Antivector, Antivector, np.ndarray, Antivector]:
    """Loading `path` scales and rotates the closed triangulated surface upright to `height` on `z = 0` alongside `n_quadrature` volume samples and `n_centers` kernel centers."""
    topology, vertex_coordinates = shapes.surface(path, height)
    V, E, F = topology.n_elements
    vertices = mv.vector(vertex_coordinates).dual().field()      # Antivector-[V]
    # Pairing each vertex with the plane y = 0 reads its height and the vertical span.
    heights = mv.y & vertices.batch()                            # [V] Scalar+
    extent = heights[heights.argmax()] - heights[heights.argmin()]   # [] Scalar+
    # Rescale to `height` along y and shift the base onto y = 0 before turning upright onto z = 0.
    vertices = vertices * (height / extent) + mv.y.dual() * (height / 2)  # Antivector-[V]
    samples, sample_volumes = volume_samples(topology, vertices, n_quadrature)  # [n_quadrature] Antivector-, [n_quadrature]
    centers = samples[center_indices(coordinates(samples), n_centers)]          # [n_centers] Antivector-
    upright_rotor = (-mv.yz * (np.pi / 4)).exp()                 # Rotor+
    return (topology, upright_rotor >> vertices.batch(), upright_rotor >> samples,
            sample_volumes, upright_rotor >> centers)
