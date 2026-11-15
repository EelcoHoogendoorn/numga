"""Spinor wave propagation and standing modes on a triangulated surface in 3D space.

Vertices carry even multivector spinors (`Even+[V]`), and the surface's sparse Dirac map (`Point-[F <- V]`) sums each corner's spinor weighted by its opposite triangle side over twice the triangle area to produce an intrinsic spinor (`Even-[F]`) on every triangle. Alternating `dirac` with its area-weighted reverse `~dirac` steps spinor waves across the surface while conserving energy, and the generalized eigenfields of `~dirac * face_areas * dirac` against `vertex_areas` give the surface's standing waves.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import NumpyContext, stack
from numga.algebras import VGA3D as ga
from numga.sparse import SparseExtensor, spdiag
from numga.topology import Topology

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                         # Scalar+
Point = ga.gatype.antivector()                      # Antivector-
Even = ga.gatype.even()                             # Even+
SpinorPair = tuple[Even, Even]


def dirac_system(topology: Topology, vertices: Point) -> tuple[SparseExtensor, SparseExtensor, SparseExtensor]:
    """The triangulated surface (`Point-[V]`) defines the sparse Dirac map (`Point-[F <- V]`) alongside the diagonal triangle-area (`Scalar+[F <- F]`) and dual-area (`Scalar+[V <- V]`) maps."""
    sides = topology.boundary[2, 1] * (topology.d[0] * vertices)                # [3] Point-[F]
    areas = sides[1].commutator(sides[0]).norm() / 2                            # Scalar+[F]
    corner_mean = topology.incidence[2, 0].mean(axis=0)                         # Scalar+[F <- V]
    dual_areas = corner_mean.adjugate() * areas                                 # Scalar+[V]
    face_areas = spdiag(areas)                                                  # Scalar+[F <- F]
    vertex_areas = spdiag(dual_areas)                                           # Scalar+[V <- V]
    dirac = (spdiag(sides / (2 * areas)) * topology.incidence[2, 0]).sum(axis=0)   # Point-[F <- V]
    return dirac, face_areas, vertex_areas


def pulse_states(
    dirac: SparseExtensor,
    face_areas: SparseExtensor,
    vertex_areas: SparseExtensor,
    initial_spinor: Even,
    dt: float,
    n_steps: int,
) -> Iterator[SpinorPair]:
    """Yields `(vertex_spinor, face_spinor)` (`Even+[V]`, `Even-[F]`) across `n_steps` leapfrog steps of step size `dt`."""
    inverse_vertex_areas = spdiag(1 / vertex_areas.diagonal())                  # Scalar+[V <- V]
    vertex_spinor = initial_spinor                                              # Even+[V]
    face_spinor = dirac * initial_spinor * 0.0                                  # Even-[F]
    for _ in range(n_steps):
        # Advance the staggered face spinor by half a step under the Dirac map.
        face_spinor = face_spinor + dirac * vertex_spinor * dt                  # Even-[F]
        yield vertex_spinor, face_spinor
        # Pull the face spinor back to the vertices through the area-weighted reverse Dirac map.
        vertex_spinor = vertex_spinor - inverse_vertex_areas * (~dirac * (face_areas * face_spinor)) * dt   # Even+[V]


def standing_modes(
    dirac: SparseExtensor,
    face_areas: SparseExtensor,
    vertex_areas: SparseExtensor,
    level_bounds: np.ndarray,
    probe: Even,
    frequencies: np.ndarray,
) -> tuple[Scalar, Even, Even]:
    """Eigenfields of `~dirac * face_areas * dirac` against `vertex_areas` determine the squared frequencies (`[n_modes] Scalar+`) and one aligned `(vertex_modes, face_modes)` pair per shell."""
    n_modes = int(level_bounds[-1])
    dirac_squared = ~dirac * face_areas * dirac                                 # Even+[V <- V]
    squared_frequencies, eigenfields = (dirac_squared * Even).eigh(vertex_areas * Even, n_modes)   # [n_modes] Scalar+, [n_modes] Even+[V]
    shells = [eigenfields[low:high] for low, high in zip(level_bounds[:-1], level_bounds[1:])]
    vertex_modes = stack([
        (shell * (vertex_areas * shell.reverse().scalar_product(probe)).sites.sum()).sum(axis=0)
        for shell in shells
    ])                                                                          # [..., ] Even+[V]
    face_modes = dirac * vertex_modes * (1 / frequencies)                       # [...] Even-[F]
    return squared_frequencies, vertex_modes, face_modes


def mode_states(vertex_modes: Even, face_modes: Even, phases: np.ndarray) -> Iterator[SpinorPair]:
    """Yields `(vertex_modes * cos(phase), face_modes * sin(phase))` across each oscillation `phase` in `phases` (`[n_phases]`)."""
    for phase in phases:
        yield vertex_modes * float(np.cos(phase)), face_modes * float(np.sin(phase))
