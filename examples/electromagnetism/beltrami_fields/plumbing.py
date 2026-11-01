"""Oriented tetrahedral meshes supply the incidences and sample geometry for Beltrami fields."""

from __future__ import annotations

import numpy as np

from numga import NumpyContext, concatenate
from numga.algebras import STA as ga
from numga.sparse import SparseExtensor, spdiag
from examples.util.topology import Topology, turned

ctx = NumpyContext(ga)
mv = ctx.multivector
Scalar = ga.gatype.scalar()  # Scalar+
Point = ga.gatype.antivector()  # Antivector-
Embedding = ga.gatype((Scalar, Scalar))
# The direction in time of an observer at rest with the torus, which pairs to one with `mv.t`.
time = -mv.t.dual()  # [] Point-


def as_scalar(values: np.ndarray) -> Scalar:
    """The supplied scalar data carries the algebra's scalar type."""
    return mv.scalar(np.asarray(values)[..., None])


def solid_torus(major_radius: float, minor_radius: float, n_major: int, n_minor: int) -> tuple[Topology, Point]:
    """A reflection-symmetric solid tetrahedral torus around the z axis."""
    major_angles = np.linspace(0.0, 2 * np.pi, n_major, endpoint=False)
    minor_angles = np.linspace(0.0, 2 * np.pi, n_minor, endpoint=False)
    around_tube = (mv.xz * (-minor_angles / 2)).exp()                    # [minor] Rotor+
    around_axis = (mv.xy * (-major_angles / 2)).exp()                    # [major] Rotor+
    unit = mv.x.dual()  # [] Point-
    rim_circle = unit * major_radius + (around_tube >> (unit * minor_radius))  # [minor] Point-
    # A cross-section, its centre on the tube's core and then its rim, turned to each step around the axis.
    section_points = concatenate(((unit * major_radius)[None], rim_circle))  # [1 + minor] Point-
    vertices = (around_axis[:, None] >> section_points[None, :]).reshape(-1).field()   # Point-[vertices]

    # The cross-section, a fan of triangles about its centre, swept around the axis: each step's prism is cut from the
    # tail of its edge, and the steps' edges run alternately backwards, so that the prisms are cut from alternate ends
    # and the solid is its own mirror image.
    around = Topology.loops([n_major]).elements[1]                          # [major, 2]
    alternating = Topology(n_major, np.where(np.arange(n_major)[:, None] % 2 == 0, around, turned(around)))
    topology = Topology.rings(1, n_minor).product(alternating)
    raw_tets = topology.elements[-1]
    # Each tetrahedron's three edges from its first corner, joined, span six times its signed volume; where that
    # volume is negative, the tetrahedron's last two corners swap.
    raw_edges = topology.oriented[3, 1] * (topology.d[0] * vertices)        # [tet_edges] Point-[tets]
    flip = ((raw_edges[0] & raw_edges[1] & raw_edges[2]) & time) < 0
    tets = raw_tets.copy()
    tets[flip, 2], tets[flip, 3] = raw_tets[flip, 3], raw_tets[flip, 2]
    return Topology(len(vertices.batch()), tets), vertices


def flux_preserving(topology: Topology, n_minor: int) -> Embedding:
    """The flux-preserving edge subspace of the solid torus."""
    # Interior edges plus the single closed poloidal boundary 1-form that carries net toroidal magnetic flux
    # while keeping the normal magnetic flux `d1(a)` identically zero on every boundary triangle.
    # Each boundary edge's step around the tube, the shorter way, from its ends' places in their cross-sections: the
    # vertices are numbered cross-section after cross-section, the centre first.
    places = topology.elements[1][topology.exterior[1]] % (n_minor + 1)     # [boundary edges, 2]
    d_pol = (places[:, 1] - places[:, 0] + n_minor // 2) % n_minor - n_minor // 2
    pol_edges = topology.exterior[1][d_pol != 0]
    pol_weights = d_pol[d_pol != 0].astype(float) / np.sqrt(np.sum(d_pol != 0))
    n_int = len(topology.interior[1])
    # The free values are one per interior edge, then the poloidal form's amplitude.
    first_values = SparseExtensor.selection(ctx, np.arange(n_int), n_int + 1)           # Scalar+[interior <- free]
    last_value = SparseExtensor.selection(ctx, np.full(len(pol_edges), n_int), n_int + 1)   # Scalar+[poloidal <- free]
    interior = ~topology.reading(topology.interior[1], 1) * first_values     # Scalar+[edges <- free]
    # The amplitude is weighted onto each boundary edge that steps around the tube.
    poloidal = ~topology.reading(pol_edges, 1) * spdiag(as_scalar(pol_weights).field()) * last_value  # Scalar+[edges <- free]
    return (interior + poloidal) * Scalar                                    # (Scalar+ <- Scalar+)[edges <- free]
