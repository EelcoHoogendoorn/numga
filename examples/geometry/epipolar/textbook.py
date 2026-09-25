"""Textbook matrix algorithms for epipolar geometry and two-view reconstruction.

This module is the matrix baseline without geometric algebra: the Hartley and Zisserman
8-point SVD, projection to the rank-2 manifold, SVD decomposition via the W matrix, testing
of the four candidate poses, and two-by-two normal equations for triangulation.

It is a point of comparison for `core`, which does the same with typed extensors and PGA3D
operators.
"""

from __future__ import annotations

import numpy as np


def essential_matrix(rays_1: np.ndarray, rays_2: np.ndarray) -> np.ndarray:
    """Solve for the Essential matrix relating two sets of corresponding sight rays.

    rays_1 and rays_2 are unit direction vectors of shape (N, 3) from Camera 1 and
    Camera 2. Each corresponding pair satisfies the epipolar coplanarity constraint:
        rays_2[k].T @ E @ rays_1[k] == 0
    """
    # Form the linear constraint matrix a of shape (N, 9), whose row k is
    # np.outer(rays_2[k], rays_1[k]).ravel():
    a = (rays_2[:, :, None] * rays_1[:, None, :]).reshape(len(rays_1), 9)

    # The optimal flattened matrix e minimizes np.linalg.norm(a @ e) subject to
    # np.linalg.norm(e) == 1 (nullspace solve):
    _, _, vh = np.linalg.svd(a)
    e_raw = vh[-1].reshape(3, 3)

    # Project onto the Essential manifold: its two non-zero singular values must be equal:
    u, _, vt = np.linalg.svd(e_raw)
    e_projected = u @ np.diag([1.0, 1.0, 0.0]) @ vt

    return e_projected


def decompose_essential(
    essential: np.ndarray,
    rays_1: np.ndarray,
    rays_2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Decompose the Essential matrix into relative rotation R and translation direction t.

    The essential matrix is the cross-product matrix of t times R, so it encodes rotation
    and translation jointly. SVD factorization yields
    four mathematically valid (R, t) pairs; the unique physical solution is selected
    by the cheirality condition (positive depth in both camera frames).
    """
    u, _, vt = np.linalg.svd(essential)

    w = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    r1 = u @ w @ vt
    r2 = u @ w.T @ vt
    if np.linalg.det(r1) < 0.0:
        r1 = -r1
    if np.linalg.det(r2) < 0.0:
        r2 = -r2
    t_candidate = u[:, 2]

    best_r = r1
    best_t = t_candidate
    max_positive_depths = -1

    for r_cand in (r1, r2):
        for sign in (1.0, -1.0):
            t_cand = sign * t_candidate
            # In Camera 2's frame: r_cand @ (d1[k] * rays_1[k]) + t_cand == d2[k] * rays_2[k].
            # Rearranged: np.stack([r_cand @ rays_1[k], -rays_2[k]], axis=-1) @ [d1[k], d2[k]] == -t_cand
            r_r1 = (r_cand @ rays_1.T).T
            m00 = np.sum(r_r1 * r_r1, axis=-1)
            m01 = np.sum(r_r1 * -rays_2, axis=-1)
            m11 = np.sum(rays_2 * rays_2, axis=-1)
            det = m00 * m11 - m01 * m01

            rhs0 = np.sum(r_r1 * -t_cand, axis=-1)
            rhs1 = np.sum(-rays_2 * -t_cand, axis=-1)

            d1 = (m11 * rhs0 - m01 * rhs1) / det
            d2 = (-m01 * rhs0 + m00 * rhs1) / det

            positive_count = int(np.sum((d1 > 0.0) & (d2 > 0.0)))
            if positive_count > max_positive_depths:
                max_positive_depths = positive_count
                best_r = r_cand
                best_t = t_cand

    return best_r, best_t


def triangulate_rays(
    c1: np.ndarray,
    rays_1: np.ndarray,
    c2: np.ndarray,
    rays_2_world: np.ndarray,
) -> np.ndarray:
    """Triangulate 3D world points from pairs of calibrated 3D sight rays.

    c1 and c2 are camera centers of shape (3,). rays_1 and rays_2_world are unit
    directions of shape (N, 3) in world coordinates. Returns the 3D midpoint of the
    shortest connecting segment between the two skew sight rays.
    """
    delta_c = c2 - c1
    r1 = rays_1
    r2 = rays_2_world

    m00 = np.sum(r1 * r1, axis=-1)
    m01 = np.sum(r1 * -r2, axis=-1)
    m11 = np.sum(r2 * r2, axis=-1)
    det = m00 * m11 - m01 * m01

    rhs0 = np.sum(r1 * delta_c, axis=-1)
    rhs1 = np.sum(-r2 * delta_c, axis=-1)

    s = (m11 * rhs0 - m01 * rhs1) / det
    u = (-m01 * rhs0 + m00 * rhs1) / det

    p1 = c1 + s[:, None] * r1
    p2 = c2 + u[:, None] * r2

    return 0.5 * (p1 + p2)


def epipolar_lines(
    points_1: np.ndarray,
    essential: np.ndarray,
) -> np.ndarray:
    """Compute 2D epipolar line coefficients in Camera 2 for points in Camera 1.

    points_1 has shape (N, 3) in normalized homogeneous coordinates (u, v, 1).
    Returns normalized line coefficients (a, b, c) where a * x + b * y + c == 0 in Camera 2.
    """
    lines = (essential @ points_1.T).T
    norm = np.linalg.norm(lines[:, :2], axis=-1, keepdims=True)
    return lines / norm
