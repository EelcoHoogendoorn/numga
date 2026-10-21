"""Triangle-mesh loading and volume quadrature for mesh-free deformation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.stats import qmc

from numga import NumpyContext
from numga.algebras import VGA3D

mv = NumpyContext(VGA3D).multivector
# Points are antivectors.
Antivector = VGA3D.gatype.antivector()
n_ray_batch = 256


def interior(points: np.ndarray, vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Classify points by the parity of their crossings with a closed triangle mesh."""
    triangles = vertices[faces]
    origin = triangles[:, 0]
    first = triangles[:, 1] - origin
    second = triangles[:, 2] - origin
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


def volume_samples(vertices: np.ndarray, faces: np.ndarray,
                   n_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Equal-volume quadrature from low-discrepancy samples inside a closed mesh."""
    lower, upper = vertices.min(axis=0), vertices.max(axis=0)
    triangles = vertices[faces]
    volume = abs(np.einsum("ij,ij->i", triangles[:, 0],
                          np.cross(triangles[:, 1], triangles[:, 2])).sum()) / 6
    occupancy = volume / np.prod(upper - lower)
    n_candidates = 2 ** int(np.ceil(np.log2(1.2 * n_samples / occupancy)))
    sequence = qmc.Sobol(3, scramble=True, seed=0)
    samples = []
    n_inside = 0
    while n_inside < n_samples:
        candidates = lower + sequence.random(n_candidates) * (upper - lower)
        accepted = candidates[interior(candidates, vertices, faces)]
        samples.append(accepted)
        n_inside += len(accepted)
    positions = np.concatenate(samples)[:n_samples]
    return positions, np.full(n_samples, volume / n_samples)


def center_indices(points: np.ndarray, n_centers: int) -> np.ndarray:
    """Farthest-point coverage of the sampled volume, including thin extremities."""
    indices = np.empty(n_centers, dtype=int)
    indices[0] = np.argmin(np.sum((points - points.mean(axis=0)) ** 2, axis=-1))
    distances = np.full(len(points), np.inf)
    for index in range(n_centers - 1):
        distances = np.minimum(distances, np.sum((points - points[indices[index]]) ** 2, axis=-1))
        indices[index + 1] = np.argmax(distances)
    return indices


def load(path: Path, height: float, n_quadrature: int,
         n_centers: int) -> tuple[Antivector, np.ndarray, Antivector, np.ndarray, Antivector]:
    """A y-up triangular OBJ, upright on z=0, with volume samples and kernel centers."""
    records = path.read_text().splitlines()
    vertices = np.array([[float(value) for value in line.split()[1:4]]
                         for line in records if line.startswith("v ")])
    faces = np.array([[int(value.split("/")[0]) - 1 for value in line.split()[1:4]]
                      for line in records if line.startswith("f ")])
    lower, upper = vertices.min(axis=0), vertices.max(axis=0)
    center = (lower + upper) / 2
    center[1] = lower[1]
    vertices = (vertices - center) * (height / (upper[1] - lower[1]))
    quadrature, weights = volume_samples(vertices, faces, n_quadrature)
    centers = quadrature[center_indices(quadrature, n_centers)]
    upright = (-mv.yz * (np.pi / 4)).exp()
    return (upright >> mv.vector(vertices).dual(), faces, upright >> mv.vector(quadrature).dual(),
            weights, upright >> mv.vector(centers).dual())
