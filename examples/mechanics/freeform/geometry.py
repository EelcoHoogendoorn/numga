"""Triangle-mesh loading and volume quadrature for mesh-free deformation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.stats import qmc

from numga import stack
from numga.algebras import VGA3D
from examples import instantiate

mesh_module = instantiate("examples.util.mesh", VGA3D)
Mesh, mv = mesh_module.Mesh, mesh_module.mv
# Points are antivectors.
Antivector = VGA3D.gatype.antivector()
n_ray_batch = 256


def coordinates(points: Antivector) -> np.ndarray:
    """Cartesian positions `[..., 3]` for the samplers, read by pairing with the planes across the axes."""
    return (stack((mv.x, mv.y, mv.z)) & points[..., None]).to_array()


def interior(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Classify points by the parity of their crossings with a closed surface of triangles, their corners
    `[3, triangles, 3]`."""
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


def volume_samples(mesh: Mesh, vertices: Antivector, n_samples: int) -> tuple[Antivector, np.ndarray]:
    """Equal-volume quadrature from low-discrepancy samples inside a closed mesh."""
    corners = mesh.incidence[2, 0] * vertices                    # [corners] Antivector-[faces]
    first, second, third = corners
    # The enclosed volume: the cones the faces span from the origin, each the join of its corners, summed.
    volume = abs((first & second & third).sites.sum().to_array()) / 6
    places = coordinates(vertices.batch())                       # [vertices, 3]
    lower, upper = places.min(axis=0), places.max(axis=0)
    triangles = coordinates(corners.batch())                     # [3, faces, 3]
    occupancy = volume / np.prod(upper - lower)
    n_candidates = 2 ** int(np.ceil(np.log2(1.2 * n_samples / occupancy)))
    sequence = qmc.Sobol(3, scramble=True, seed=0)
    samples = []
    n_inside = 0
    while n_inside < n_samples:
        candidates = lower + sequence.random(n_candidates) * (upper - lower)
        accepted = candidates[interior(candidates, triangles)]
        samples.append(accepted)
        n_inside += len(accepted)
    positions = np.concatenate(samples)[:n_samples]
    return mv.vector(positions).dual(), np.full(n_samples, volume / n_samples)


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
         n_centers: int) -> tuple[Mesh, Antivector, Antivector, np.ndarray, Antivector]:
    """A y-up triangular OBJ, upright on z=0 and `height` tall, its vertices a batch of points for the
    deformation to move, with volume samples and kernel centers."""
    mesh, vertices = Mesh.read(path, height)                     # Antivector-[vertices]
    # Each vertex's height, its pairing with the plane y = 0, and the extent of the heights.
    heights = mv.y & vertices.batch()                            # [vertices] Scalar+
    extent = heights[heights.argmax()] - heights[heights.argmin()]   # [] Scalar+
    # As tall as `height` along y, its lowest point lifted onto the plane y = 0.
    vertices = vertices * (height / extent) + mv.y.dual() * (height / 2)   # Antivector-[vertices]
    samples, weights = volume_samples(mesh, vertices, n_quadrature)   # [samples] Antivector-, [samples]
    centers = samples[center_indices(coordinates(samples), n_centers)]   # [centers] Antivector-
    upright = (-mv.yz * (np.pi / 4)).exp()
    return mesh, upright >> vertices.batch(), upright >> samples, weights, upright >> centers
