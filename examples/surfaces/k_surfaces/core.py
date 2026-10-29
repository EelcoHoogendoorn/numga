"""Discrete K-surfaces grow from compatible reflections of their tangent planes.

The normals determine edge displacements whose sums close around every quadrilateral.
An extra direction of the same construction gives a Bäcklund partner: corresponding
points are joined by equal-length segments tangent to both surfaces. Completing two
such transformations gives the same surface in either order.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from numga import NumpyContext, concatenate
from numga.algebras import VGA3D as ga

mv = NumpyContext(ga).multivector
Normal = ga.gatype.vector()                         # Vector+
Point = ga.gatype.antivector()                      # Antivector-
Antipseudoscalar = ga.gatype.antipseudoscalar()   # Antipseudoscalar-


# --- math -----------------------------------------------------------------------------
def complete(base: Normal, left: Normal, right: Normal) -> Normal:
    """The missing normal of a quadrilateral is its opposite normal reflected in the plane bisecting its two
    neighbours."""
    return (left - right).normalized() >> base


def positions(normals: Normal, radius: Antipseudoscalar, origin: Point) -> Point:
    """The surface's points sum the edges between neighbours, the lines where their tangent planes cross, from the
    first grid point at the origin."""
    # Each normal gives a map from a neighbouring normal to a tangent displacement.
    edge_map = radius[..., None, None] * Normal.commutator(normals)      # [..., rows, columns] Point- <- Normal+
    edges_u = edge_map[..., :-1, :](normals[..., 1:, :])                  # [..., rows - 1, columns] Antivector-
    edges_v = edge_map[..., :, :-1](normals[..., :, 1:])                  # [..., rows, columns - 1] Antivector-
    first = origin + edges_u[..., 0, 0] * 0
    row_origins = concatenate([first[..., None], first[..., None] + edges_u[..., :, 0].cumsum(axis=-1)], axis=-1)
    return row_origins[..., :, None] + concatenate([edges_v[..., :, :1] * 0, edges_v.cumsum(axis=-1)], axis=-1)


# --- plumbing -------------------------------------------------------------------------
def normal_net(mid_u: Normal, mid_v: Normal, completion: Callable[[Normal, Normal, Normal], Normal]) -> Normal:
    """Two crossing odd-length normal sequences extend over their rectangular grid."""
    n_u, n_v = mid_u.shape[-1], mid_v.shape[-1]
    n_half_u, n_half_v = n_u // 2, n_v // 2
    crossing = mid_u[..., n_half_u] + mid_v[..., n_half_v]
    normals = crossing[..., None, None] * np.zeros((n_u, n_v))           # [..., rows, columns] Vector+
    normals = normals.at[..., :, n_half_v].set(mid_u)
    normals = normals.at[..., n_half_u, :].set(mid_v)
    directions_u = np.array([1, 1, -1, -1])[:, None]
    directions_v = np.array([1, -1, 1, -1])[:, None]
    # Each diagonal depends only on the two preceding diagonals, in all four quadrants.
    for diagonal in range(2, n_half_u + n_half_v + 1):
        distance_u = np.arange(max(1, diagonal - n_half_v), min(n_half_u, diagonal - 1) + 1)
        distance_v = diagonal - distance_u
        row = n_half_u + directions_u * distance_u
        column = n_half_v + directions_v * distance_v
        previous_row = row - directions_u
        previous_column = column - directions_v
        normals = normals.at[..., row, column].set(completion(
            normals[..., previous_row, previous_column],
            normals[..., row, previous_column],
            normals[..., previous_row, column]))
    return normals


def partner(normals: Normal, start: Normal, completion: Callable[[Normal, Normal, Normal], Normal]) -> Normal:
    """A Bäcklund normal field propagates from its value at the first grid point."""
    n_u, n_v = normals.shape[-2:]
    result = (normals + start[..., None, None]) * 0                      # [..., rows, columns] Vector+
    result = result.at[..., 0, 0].set(start)
    for row in range(1, n_u):
        result = result.at[..., row, 0].set(completion(
            normals[..., row - 1, 0], normals[..., row, 0], result[..., row - 1, 0]))
    # With the first column known, every row advances independently across the grid.
    for column in range(1, n_v):
        result = result.at[..., :, column].set(completion(
            normals[..., :, column - 1], normals[..., :, column], result[..., :, column - 1]))
    return result
