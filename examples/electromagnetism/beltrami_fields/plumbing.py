"""Oriented tetrahedral meshes supply the incidences and sample geometry for Beltrami fields."""

from __future__ import annotations

import numpy as np

from numga import NumpyContext, concatenate
from numga.algebras import STA as ga
from numga.sparse import SparseExtensor, spdiag
from examples import instantiate

mesh_module = instantiate("examples.util.mesh", ga)
Mesh, simplex_parity = mesh_module.Mesh, mesh_module.simplex_parity
ctx = NumpyContext(ga)
mv = ctx.multivector
Scalar = ga.gatype.scalar()  # Scalar+
Point = ga.gatype.antivector()  # Antivector-
Embedding = ga.gatype((Scalar, Scalar))
# The direction in time of an observer at rest with the torus, which pairs to one with `mv.t`.
time = -mv.t.dual()  # [] Point-


def complements(mesh: Mesh, m: int) -> np.ndarray:
    """`[k, 3 - m]`: the corners each of a tet's m-simplices leaves out, in the order of `mesh.corners(3, m)`, run so
    that the m-simplex followed by them runs the tet's own sense: facing each corner, its face, outward; across each
    edge, the opposite edge."""
    simplices = mesh.corners(3, m)                                           # [k, m + 1]
    tet = mesh.corners(3, 0)[:, 0]                                           # [corners]
    left_out = np.stack([np.setdiff1d(tet, simplex) for simplex in simplices])   # [k, 3 - m]
    sense = simplex_parity(np.concatenate([simplices, left_out], axis=-1))   # [k]
    return np.where(sense[:, None] > 0, left_out, left_out[:, ::-1])


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
    raw_mesh = Mesh(len(vertices.batch()), raw_tets)
    # Each tetrahedron's three edges from its first corner, joined, span six times its signed volume; where that
    # volume is negative, the tetrahedron's last two corners swap.
    raw_edges = raw_mesh.oriented[3, 1] * (raw_mesh.d[0] * vertices)        # [tet_edges] Point-[tets]
    flip = ((raw_edges[0] & raw_edges[1] & raw_edges[2]) & time) < 0
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
    # The free values are one per interior edge, then the poloidal form's amplitude.
    first_values = SparseExtensor.selection(ctx, np.arange(n_int), n_int + 1)           # Scalar+[interior <- free]
    last_value = SparseExtensor.selection(ctx, np.full(len(pol_edges), n_int), n_int + 1)   # Scalar+[poloidal <- free]
    interior = ~mesh.reading(mesh.interior_edges, 1) * first_values   # Scalar+[edges <- free]
    # The amplitude is weighted onto each boundary edge that steps around the tube.
    poloidal = ~mesh.reading(pol_edges, 1) * spdiag(as_scalar(pol_weights).field()) * last_value   # Scalar+[edges <- free]
    return (interior + poloidal) * Scalar                                    # (Scalar+ <- Scalar+)[edges <- free]
