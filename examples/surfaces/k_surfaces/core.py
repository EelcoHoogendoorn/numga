"""Discrete K-surfaces grow from compatible reflections of their tangent planes.

Two crossing curves of unit normals determine a rectangular net of normals by reflecting each quadrilateral's base normal across the bisector of its two adjacent normals. Neighboring tangent planes cross along edge displacements that close around every quadrilateral, and propagating the same reflection into an extra grid direction produces a Bäcklund surface joined to the original sheet by equal-length tangent segments.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import numpy as np

from numga import NumpyContext, concatenate
from numga.algebras import VGA3D as ga

mv = NumpyContext(ga).multivector
Plane = ga.gatype.vector()                          # Vector+
Point = ga.gatype.antivector()                      # Antivector-
Rotor = ga.gatype.rotor()                           # Rotor+
Antiscalar = ga.gatype.antipseudoscalar()           # Antipseudoscalar-


# --- math -----------------------------------------------------------------------------
def quad_normal(base: Plane, left: Plane, right: Plane) -> Plane:
    """Reflecting `base` across the bisector plane `(left - right).normalized()` completes a compatible quadrilateral of unit normals (`[...] Plane+`)."""
    bisector = (left - right).normalized()                                      # [...] Plane+
    return bisector >> base                                                     # [...] Plane+


def surface_points(normals: Plane, radius: Antiscalar, origin: Point) -> Point:
    """Summing tangent-plane intersection edges `radius * Plane.commutator(normals)(neighbour)` from `origin` reconstructs the K-surface vertices (`[..., n_u, n_v] Point-`)."""
    n_u, n_v = normals.shape[-2:]
    del n_u, n_v
    # Each normal maps a neighboring normal to the displacement along the line where their tangent planes meet.
    edge_map = radius[..., None, None] * Plane.commutator(normals)              # [..., n_u, n_v] Point- <- Plane+
    edges_u = edge_map[..., :-1, :](normals[..., 1:, :])                        # [..., n_u - 1, n_v] Point-
    edges_v = edge_map[..., :, :-1](normals[..., :, 1:])                        # [..., n_u, n_v - 1] Point-
    first = origin + edges_u[..., 0, 0] * 0                                     # [...] Point-
    row_origins = concatenate([first[..., None], first[..., None] + edges_u[..., :, 0].cumsum(axis=-1)], axis=-1)
    return row_origins[..., :, None] + concatenate([edges_v[..., :, :1] * 0, edges_v.cumsum(axis=-1)], axis=-1)


def normal_net(mid_u: Plane, mid_v: Plane, completion: Callable[[Plane, Plane, Plane], Plane]) -> Plane:
    """Two crossing odd-length normal curves `mid_u` (`[..., n_u] Plane+`) and `mid_v` (`[..., n_v] Plane+`) propagate diagonally across the `[..., n_u, n_v]` net."""
    n_u, n_v = mid_u.shape[-1], mid_v.shape[-1]
    n_half_u, n_half_v = n_u // 2, n_v // 2
    crossing = mid_u[..., n_half_u] + mid_v[..., n_half_v]                      # [...] Plane+
    normals = crossing[..., None, None] * np.zeros((n_u, n_v))                  # [..., n_u, n_v] Plane+
    normals = normals.at[..., :, n_half_v].set(mid_u)
    normals = normals.at[..., n_half_u, :].set(mid_v)
    directions_u = np.array([1, 1, -1, -1])[:, None]
    directions_v = np.array([1, -1, 1, -1])[:, None]
    # Each diagonal in all four quadrants depends only on the two preceding diagonals.
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


def backlund_normals(normals: Plane, start: Plane, completion: Callable[[Plane, Plane, Plane], Plane]) -> Plane:
    """Propagating `completion` from `start` at corner `(0, 0)` builds a compatible Bäcklund normal net (`[..., n_u, n_v] Plane+`)."""
    n_u, n_v = normals.shape[-2:]
    result = (normals + start[..., None, None]) * 0                             # [..., n_u, n_v] Plane+
    result = result.at[..., 0, 0].set(start)
    for row in range(1, n_u):
        result = result.at[..., row, 0].set(completion(
            normals[..., row - 1, 0], normals[..., row, 0], result[..., row - 1, 0]))
    # Once the first column is determined, every row advances in lockstep across the columns.
    for column in range(1, n_v):
        result = result.at[..., :, column].set(completion(
            normals[..., :, column - 1], normals[..., :, column], result[..., :, column - 1]))
    return result


def saddle_patch(n_half: int, extent: float, corner_angle: float, radius: Antiscalar) -> tuple[Plane, Point]:
    """Two great-circle normal arcs meeting at `corner_angle` generate a centred K-surface patch (`[n_vertices, n_vertices] Plane+`, `[n_vertices, n_vertices] Point-`)."""
    n_vertices = 2 * n_half + 1
    angles = np.linspace(-extent, extent, n_vertices)                           # [n_vertices]
    alternating = 1 - 2 * ((np.arange(n_vertices) - n_half) % 2)                # [n_vertices]
    reference_normal = mv.vector([0, 0, 1])                                     # [] Plane+
    corner_turn = (mv.xy * (-corner_angle / 2)).exp()                           # [] Rotor+
    second_plane = corner_turn >> mv.yz                                         # [] Bivector+
    turns_u = (mv.yz * (-angles / 2)).exp()                                     # [n_vertices] Rotor+
    mid_u = turns_u >> reference_normal                                         # [n_vertices] Plane+
    turns_v = (second_plane * (-angles / 2)).exp()                              # [n_vertices] Rotor+
    # Alternating signs on adjacent columns allow one reflection rule to serve both grid directions.
    mid_v = alternating * (turns_v >> reference_normal)                         # [n_vertices] Plane+
    normals = normal_net(mid_u, mid_v, quad_normal)                             # [n_vertices, n_vertices] Plane+
    points = surface_points(normals, radius, reference_normal.dual() * 0)       # [n_vertices, n_vertices] Point-
    return normals, points - points[n_half, n_half]


def saddle_states(n_half: int, extent: float, corner_angles: np.ndarray, radius: Antiscalar) -> Iterator[Point]:
    """Yields centred saddle vertex nets (`[n_vertices, n_vertices] Point-`) as the central normal-arc angle sweeps `corner_angles`."""
    for corner_angle in corner_angles:
        normals, points = saddle_patch(n_half, extent, float(corner_angle), radius)
        del normals
        yield points
