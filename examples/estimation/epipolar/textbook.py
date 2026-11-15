"""Eight-point matrix baseline computes the essential matrix, four-way pose decomposition, midpoint triangulation, and epipolar lines without geometric algebra.

Hartley and Zisserman's singular-value solve fits an unconstrained three-by-three matrix to the bilinear ray constraints, projects onto the equal-singular-value manifold, factors four candidate rotation-translation pairs, selects the physical pair by positive depth counts in both camera frames, and triangulates skew sight rays through two-by-two normal equations.
"""

from __future__ import annotations

import numpy as np


def essential_matrix(rays_1: np.ndarray, rays_2: np.ndarray) -> np.ndarray:
    """Two singular value decompositions fit a rank-two essential matrix (`[3, 3]`) with equal nonzero singular values to calibrated sight-ray directions `rays_1` and `rays_2` (`[n_rays, 3]`)."""
    # Each row flattens the outer product of one corresponding sight-ray pair so the bilinear constraint reads as design @ e == 0.
    design = (rays_2[:, :, None] * rays_1[:, None, :]).reshape(len(rays_1), 9)
    _, _, right_singular = np.linalg.svd(design)
    raw_essential = right_singular[-1].reshape(3, 3)

    # Enforcing equal nonzero singular values projects the unconstrained least-squares matrix onto the essential manifold.
    left_factors, _, right_factors = np.linalg.svd(raw_essential)
    return left_factors @ np.diag([1.0, 1.0, 0.0]) @ right_factors


def candidate_pose(
    essential: np.ndarray,
    rays_1: np.ndarray,
    rays_2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Singular-value factorization of `essential` (`[3, 3]`) tests four candidate poses and selects the rotation (`[3, 3]`) and unit translation (`[3]`) giving positive depths along `rays_1` and `rays_2` (`[n_rays, 3]`)."""
    left_factors, _, right_factors = np.linalg.svd(essential)

    quarter_turn = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    rotation_a = left_factors @ quarter_turn @ right_factors
    rotation_b = left_factors @ quarter_turn.T @ right_factors
    if np.linalg.det(rotation_a) < 0.0:
        rotation_a = -rotation_a
    if np.linalg.det(rotation_b) < 0.0:
        rotation_b = -rotation_b
    translation_axis = left_factors[:, 2]

    chosen_rotation = rotation_a
    chosen_translation = translation_axis
    max_positive_depths = -1

    for candidate_rotation in (rotation_a, rotation_b):
        for sign in (1.0, -1.0):
            candidate_translation = sign * translation_axis
            # Solving the 2x2 normal equations for ray depths in camera 2's frame counts points lying in front of both cameras.
            rotated_rays_1 = (candidate_rotation @ rays_1.T).T
            gram_00 = np.sum(rotated_rays_1 * rotated_rays_1, axis=-1)
            gram_01 = np.sum(rotated_rays_1 * -rays_2, axis=-1)
            gram_11 = np.sum(rays_2 * rays_2, axis=-1)
            determinant = gram_00 * gram_11 - gram_01 * gram_01

            rhs_0 = np.sum(rotated_rays_1 * -candidate_translation, axis=-1)
            rhs_1 = np.sum(-rays_2 * -candidate_translation, axis=-1)

            depths_1 = (gram_11 * rhs_0 - gram_01 * rhs_1) / determinant
            depths_2 = (-gram_01 * rhs_0 + gram_00 * rhs_1) / determinant

            positive_count = int(np.sum((depths_1 > 0.0) & (depths_2 > 0.0)))
            if positive_count > max_positive_depths:
                max_positive_depths = positive_count
                chosen_rotation = candidate_rotation
                chosen_translation = candidate_translation

    return chosen_rotation, chosen_translation


def ray_midpoints(
    centre_1: np.ndarray,
    rays_1: np.ndarray,
    centre_2: np.ndarray,
    rays_2_world: np.ndarray,
) -> np.ndarray:
    """Midpoints (`[n_rays, 3]`) of the shortest segments between skew world-frame rays from `centre_1` along `rays_1` and `centre_2` along `rays_2_world` triangulate the 3D landmarks."""
    baseline = centre_2 - centre_1

    gram_00 = np.sum(rays_1 * rays_1, axis=-1)
    gram_01 = np.sum(rays_1 * -rays_2_world, axis=-1)
    gram_11 = np.sum(rays_2_world * rays_2_world, axis=-1)
    determinant = gram_00 * gram_11 - gram_01 * gram_01

    rhs_0 = np.sum(rays_1 * baseline, axis=-1)
    rhs_1 = np.sum(-rays_2_world * baseline, axis=-1)

    depths_1 = (gram_11 * rhs_0 - gram_01 * rhs_1) / determinant
    depths_2 = (-gram_01 * rhs_0 + gram_00 * rhs_1) / determinant

    points_1 = centre_1 + depths_1[:, None] * rays_1
    points_2 = centre_2 + depths_2[:, None] * rays_2_world

    return 0.5 * (points_1 + points_2)


def epipolar_lines(
    points_1: np.ndarray,
    essential: np.ndarray,
) -> np.ndarray:
    """Multiplying normalized homogeneous image points `points_1` (`[n_points, 3]`) by `essential` (`[3, 3]`) gives Euclidean-normalized epipolar line coefficients (`[n_points, 3]`) in camera 2."""
    lines = (essential @ points_1.T).T
    slope_norm = np.linalg.norm(lines[:, :2], axis=-1, keepdims=True)
    return lines / slope_norm
