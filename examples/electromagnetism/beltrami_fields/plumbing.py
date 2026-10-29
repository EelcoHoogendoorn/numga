"""Oriented tetrahedral meshes supply the incidences and sample geometry for Beltrami fields."""

from __future__ import annotations

import numpy as np

from numga import NumpyContext, concatenate
from numga.algebras import STA as ga
from numga.sparse import SparseExtensor
from examples import instantiate

Mesh = instantiate("examples.util.mesh", ga).Mesh
ctx = NumpyContext(ga)
mv = ctx.multivector
Scalar = ga.gatype.scalar()  # Scalar+
Point = ga.gatype.antivector()  # Antivector-
Embedding = ga.gatype((Scalar, Scalar))
# The direction in time of an observer at rest with the torus, which pairs to one with `mv.t`.
time = -mv.t.dual()  # [] Point-

# Local corner indices of the six edges (0,1), (0,2), (0,3), (1,2), (1,3), (2,3),
# their positively oriented opposite edges, and the four outward opposite faces.
edge_tails = np.array([0, 0, 0, 1, 1, 2])
edge_heads = np.array([1, 2, 3, 2, 3, 3])
opposite_tails = np.array([2, 3, 1, 0, 2, 0])
opposite_heads = np.array([3, 1, 2, 3, 0, 1])
face_first = np.array([1, 0, 0, 0])
face_second = np.array([2, 3, 1, 2])
face_third = np.array([3, 2, 3, 1])


def as_scalar(values: np.ndarray) -> Scalar:
    """The supplied scalar data carries the algebra's scalar type."""
    return mv.scalar(np.asarray(values)[..., None])


def solid_torus(major_radius: float, minor_radius: float, n_major: int, n_minor: int) -> tuple[Mesh, Point]:
    """A reflection-symmetric solid tetrahedral torus around the z axis."""
    major_angles = np.linspace(0.0, 2 * np.pi, n_major, endpoint=False)
    minor_angles = np.linspace(0.0, 2 * np.pi, n_minor, endpoint=False)
    around_tube = (mv.xz * (-minor_angles / 2)).exp()                    # [minor] Rotor+
    around_axis = (mv.xy * (-major_angles / 2)).exp()                    # [major] Rotor+
    unit = mv.x.dual()                                                   # [] Point-
    core_circle = around_axis >> (unit * major_radius)                   # [major] Point-
    rim_circle = unit * major_radius + (around_tube >> (unit * minor_radius))  # [minor] Point-
    rim_points = (around_axis[:, None] >> rim_circle[None, :]).reshape(-1)  # [major * minor] Point-
    vertices = concatenate((core_circle, rim_points)).field()            # Point-[vertices]

    v = np.arange(n_minor)
    v_next = (v + 1) % n_minor
    cross_sorted = np.sort(np.stack([np.zeros(n_minor, dtype=int), 1 + v, 1 + v_next], axis=-1), axis=-1)

    def slice_vertex(u_idx: np.ndarray, local_v: np.ndarray) -> np.ndarray:
        return np.where(local_v == 0, u_idx, n_major + u_idx * n_minor + (local_v - 1))

    u = np.arange(n_major)[:, None]
    u_next = (u + 1) % n_major
    even = u % 2 == 0
    u0, u1 = np.where(even, u, u_next), np.where(even, u_next, u)
    a, b, d = cross_sorted[:, 0][None, :], cross_sorted[:, 1][None, :], cross_sorted[:, 2][None, :]
    a0, b0, d0 = slice_vertex(u0, a), slice_vertex(u0, b), slice_vertex(u0, d)
    a1, b1, d1 = slice_vertex(u1, a), slice_vertex(u1, b), slice_vertex(u1, d)
    raw_tets = np.concatenate([
        np.stack([a0, b0, d0, d1], axis=-1).reshape(-1, 4),
        np.stack([a0, b0, d1, b1], axis=-1).reshape(-1, 4),
        np.stack([a0, b1, d1, a1], axis=-1).reshape(-1, 4),
    ], axis=0)
    corners_raw = vertices.batch()[raw_tets.T]
    vols_raw = ((corners_raw[1] - corners_raw[0]) & (corners_raw[2] - corners_raw[0]) & (corners_raw[3] - corners_raw[0])) / 6
    flip = (vols_raw & time) < 0
    tets = raw_tets.copy()
    tets[flip, 2], tets[flip, 3] = raw_tets[flip, 3], raw_tets[flip, 2]
    return Mesh(len(vertices.batch()), tets), vertices


def flux_preserving(mesh: Mesh, n_major: int, n_minor: int) -> Embedding:
    """The flux-preserving edge subspace of the solid torus."""
    # Interior edges plus the single closed poloidal boundary 1-form that carries net toroidal magnetic flux
    # while keeping the normal magnetic flux `d1(a)` identically zero on every boundary triangle.
    b_edges = mesh.edges[mesh.boundary_edges]
    d_pol = ((b_edges[:, 1] - n_major) % n_minor - (b_edges[:, 0] - n_major) % n_minor + n_minor // 2) % n_minor - n_minor // 2
    pol_edges = mesh.boundary_edges[d_pol != 0]
    pol_weights = d_pol[d_pol != 0].astype(float) / np.sqrt(np.sum(d_pol != 0))
    n_int = len(mesh.interior_edges)
    rows = np.concatenate([mesh.interior_edges, pol_edges])
    cols = np.concatenate([np.arange(n_int), np.full(len(pol_edges), n_int)])
    vals = np.concatenate([np.ones(n_int), pol_weights])
    return SparseExtensor.from_indices(as_scalar(vals), rows, cols, (len(mesh.edges), n_int + 1)) * Scalar  # (Scalar+ <- Scalar+)[edges <- free]
