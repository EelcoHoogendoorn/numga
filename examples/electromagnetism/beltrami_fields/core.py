"""Tetrahedral Whitney forms carry magnetic energy, topological helicity, and Beltrami fields.

In spacetime, an observer at rest with the toroidal cavity has worldline tangent `time`, spatial positions are intrinsic points, and a static magnetic field is a spatial bivector `Field` whose flux through an oriented surface patch reads as `field & patch`. Each corner's simplex gradient sweeps its opposite outward face patch along `time` over three times the swept tetrahedral prism `volumes & time`, building Whitney edge 1-forms and their curls without a metric. Magnetic energy pulls the vacuum duality map `Field.dual()` back through `curl`, whereas magnetic helicity wedges the potential 1-form with signed volume and pulls the pairing back through `curl`.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import Algebra, NumpyContext
from numga.algebras import STA as ga
from numga.sparse import spdiag
from numga.topology import Topology

ctx = NumpyContext(ga)
mv = ctx.multivector
Scalar = ga.gatype.scalar()                                 # Scalar+
Plane = ga.gatype.vector()                                  # Vector+
SpatialPlane = Plane.derive.within(Algebra("x+y+z+"))       # Vector+
# Static magnetic fields lie in the spatial bivector planes of the cavity rest frame.
Field = ga.gatype.bivector().derive.within(Algebra("x+y+z+"))  # Bivector+
Point = ga.gatype.antivector()                              # Antivector-
Volume = ga.gatype.antitrivector()                          # Antitrivector-
Antiscalar = ga.gatype.antiscalar()                         # Antiscalar-
OneForm = ga.gatype((SpatialPlane, Scalar))                 # Plane+ <- Scalar+
Curl = ga.gatype((Field, Scalar))                           # Field+ <- Scalar+
Form = ga.gatype((Antiscalar, Scalar))                      # Antiscalar- <- Scalar+

# Worldline tangent of an observer at rest with the torus, pairing to one with `mv.t`.
time = -mv.t.dual()                                         # [] Point-


def simplex_volumes(topology: Topology, vertices: Point) -> Volume:
    """The three edges from each tetrahedron's first corner span six times its signed volume (`Volume-[T]`)."""
    edges = topology.oriented[3, 1] * (topology.d[0] * vertices)                # [6] Point-[T]
    return (edges[0] & edges[1] & edges[2]) / 6                                 # Volume-[T]


def simplex_gradients(topology: Topology, vertices: Point, volumes: Volume) -> SpatialPlane:
    """Sweeping each outward face patch along `time` gives the four corner simplex gradients (`[4] Plane+[T]`)."""
    sides = topology.boundary[2, 1] * (topology.d[0] * vertices)                # [3] Point-[F]
    face_patches = topology.boundary[3, 2] * ((sides[0] & sides[1]) / 2)        # [4] Patch-[T]
    prisms = volumes & time                                                     # Antiscalar-[T]
    return (face_patches & time) / (3 * prisms)                                 # [4] Plane+[T]


def whitney_1form(topology: Topology, gradients: SpatialPlane) -> OneForm:
    """Each edge's Whitney 1-form evaluates to one quarter of its endpoint gradient difference at centroids (`(Plane+ <- Scalar+)[T <- E]`)."""
    # At each tetrahedron centroid the four corner weights equal 1/4, reducing the edge 1-form
    # to one quarter of the difference of its endpoints' simplex gradients.
    tails, heads = topology.corners[3, 1]
    centroid_1forms = (gradients[heads] - gradients[tails]) / 4                 # [6] Plane+[T]
    return (spdiag(centroid_1forms * Scalar) * topology.oriented[3, 1]).sum(axis=0)  # (Plane+ <- Scalar+)[T <- E]


def whitney_curl(topology: Topology, gradients: SpatialPlane) -> Curl:
    """The exterior derivative of the edge Whitney 1-forms gives the magnetic field map (`(Field+ <- Scalar+)[T <- E]`)."""
    # Each edge's Whitney curl is twice the wedge of its endpoints' simplex gradients, assembled
    # with the relative edge orientations `topology.oriented[3, 1]`.
    tails, heads = topology.corners[3, 1]
    curls = 2 * (gradients[tails] ^ gradients[heads])                           # [6] Field+[T]
    return (spdiag(curls * Scalar) * topology.oriented[3, 1]).sum(axis=0)       # (Field+ <- Scalar+)[T <- E]


def magnetic_energy(volumes: Volume, curl: Curl) -> Form:
    """Pulling the volume-weighted vacuum duality map back through `curl` gives the magnetic energy form (`(Antiscalar- <- Scalar+)[E <- E]`)."""
    vacuum = (volumes & time).norm() * Field.dual()                             # (Excitation- <- Field+)[T]
    return curl.adjugate()(vacuum(curl))                                        # (Antiscalar- <- Scalar+)[E <- E]


def magnetic_helicity(volumes: Volume, potential_map: OneForm, curl: Curl) -> Form:
    """Wedging `potential_map` with `volumes` and pulling back through `curl` gives the symmetrized helicity form (`(Antiscalar- <- Scalar+)[E <- E]`)."""
    volume_weights = Plane ^ volumes                                            # (Excitation- <- Plane+)[T]
    raw_helicity = curl.adjugate()(volume_weights(potential_map))               # (Antiscalar- <- Scalar+)[E <- E]
    return (raw_helicity + raw_helicity.adjugate()) / 2                         # (Antiscalar- <- Scalar+)[E <- E]


def relaxation_states(
    initial_amplitudes: Scalar,
    eigenvalues: Scalar,
    modes: Scalar,
    curl: Curl,
    times: np.ndarray,
) -> Iterator[tuple[Scalar, Field]]:
    """Yields modal amplitudes and reconstructed magnetic fields (`[...] Scalar+`, `Field+[T]`) under helicity-preserving decay."""
    initial_helicity = (initial_amplitudes.squared() / eigenvalues).sum(axis=0)  # [] Scalar+
    for decay_time in times:
        # Finer modes decay faster; the shared scale fixes the linkage of the whole field.
        raw_amplitudes = initial_amplitudes * (-eigenvalues.squared() * decay_time).exp()  # [...] Scalar+
        raw_helicity = (raw_amplitudes.squared() / eigenvalues).sum(axis=0)                # [] Scalar+
        amplitudes = raw_amplitudes * (initial_helicity / raw_helicity).square_root()      # [...] Scalar+
        cochain = (amplitudes * modes).sum(axis=0)                                         # Scalar+[E]
        yield amplitudes, curl(cochain)
