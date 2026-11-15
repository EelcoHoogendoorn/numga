"""Tetrahedral torus topologies supply the incidences and flux-preserving edge subspaces for Beltrami fields."""

from __future__ import annotations

import numpy as np

from numga import concatenate
from numga.sparse import Sparse
from numga.topology import Topology, reverse_orientation
from examples.electromagnetism.beltrami_fields.core import Point, mv, time


def solid_torus(major_radius: float, minor_radius: float, n_major: int, n_minor: int) -> tuple[Topology, Point]:
    """Sweeping a triangulated disc around the vertical axis builds a reflection-symmetric tetrahedral torus (`Point-[V]`)."""
    major_angles = np.linspace(0.0, 2 * np.pi, n_major, endpoint=False)         # [n_major]
    minor_angles = np.linspace(0.0, 2 * np.pi, n_minor, endpoint=False)         # [n_minor]
    tube_rotors = (mv.xz * (-minor_angles / 2)).exp()                           # [n_minor] Rotor+
    axis_rotors = (mv.xy * (-major_angles / 2)).exp()                           # [n_major] Rotor+
    radial_unit = mv.x.dual()                                                   # [] Point-
    minor_rim = tube_rotors >> (radial_unit * minor_radius)                     # [n_minor] Point-
    rim_circle = radial_unit * major_radius + minor_rim                         # [n_minor] Point-
    # Each cross-section places its centre on the tube's core and then its rim, turned around the vertical axis.
    section_points = concatenate(((radial_unit * major_radius)[None], rim_circle))  # [n_minor + 1] Point-
    section_grid = axis_rotors[:, None] >> section_points[None, :]              # [n_major, n_minor + 1] Point-
    vertices = section_grid.reshape(-1).field()                                 # Point-[V]

    # Alternating ring edge orientations cut adjacent prisms from opposite ends so the solid is reflection-symmetric.
    loop_edges = Topology.loops([n_major]).elements[1]                          # [n_major, 2]
    alternating = Topology(
        n_major,
        np.where(np.arange(n_major)[:, None] % 2 == 0, loop_edges, reverse_orientation(loop_edges)),
    )
    topology = Topology.rings(1, n_minor).product(alternating)
    V, E, F, T = topology.n_elements
    raw_tets = topology.elements[3]                                             # [T, 4]
    # Swapping the last two corners where the triple join with `time` is negative orients every tetrahedron positively.
    raw_edges = topology.oriented[3, 1] * (topology.d[0] * vertices)            # [6] Point-[T]
    negative_volume = ((raw_edges[0] & raw_edges[1] & raw_edges[2]) & time) < 0
    tets = raw_tets.copy()
    tets[negative_volume, 2], tets[negative_volume, 3] = raw_tets[negative_volume, 3], raw_tets[negative_volume, 2]
    return Topology(V, tets), vertices


def flux_preserving(topology: Topology, n_minor: int) -> Sparse:
    """The interior edges and closed poloidal boundary circulation embed into the full edge space (`Scalar+[E <- E_free]`)."""
    # Interior edges plus one closed poloidal boundary 1-form carry net toroidal magnetic flux
    # while keeping normal magnetic flux identically zero on every boundary triangle.
    V, E, F, T = topology.n_elements
    places = topology.elements[1][topology.exterior[1]] % (n_minor + 1)         # [E_exterior, 2]
    poloidal_steps = (places[:, 1] - places[:, 0] + n_minor // 2) % n_minor - n_minor // 2
    poloidal_edges = topology.exterior[1][poloidal_steps != 0]
    poloidal_weights = poloidal_steps[poloidal_steps != 0].astype(float) / np.sqrt(np.sum(poloidal_steps != 0))
    n_interior = len(topology.interior[1])
    n_free = n_interior + 1
    interior = Sparse.from_indices(np.ones(n_interior), topology.interior[1], np.arange(n_interior), (E, n_free))
    poloidal = Sparse.from_indices(poloidal_weights, poloidal_edges, np.full(len(poloidal_edges), n_interior), (E, n_free))
    return interior + poloidal                                                  # Scalar+[E <- E_free]
