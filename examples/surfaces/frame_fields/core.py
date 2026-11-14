"""Rotations attached to the vertices of a triangulated surface interpolate into a continuous field of orientations. Blending corner rotors linearly traces a chord inside the unit rotor sphere whose changing orientation and changing length occupy grades two and zero of one derivative.

The Darboux map `2 * chord_rotors.reverse() * rotor_differential` takes any displacement across a triangle to that mixed-grade change, expressing the turning plane directly in Whitney's edge 1-forms. This module follows the local triangle decomposition into a planar disk that wraps the sphere of directors once and into harmonic and unit-constrained frame fields on a curved saddle surface.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import NumpyContext
from numga.algebras import VGA3D as ga
from numga.sparse import SparseExtensor, spdiag
from numga.topology import Topology

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                                                  # Scalar+
Plane = ga.gatype.vector()                                                   # Vector+
Bivector = ga.gatype.bivector()                                              # Bivector+
Even = ga.gatype.even()                                                      # Even+
Rotor = ga.gatype.rotor()                                                    # Rotor+
Point = ga.gatype.antivector()                                               # Antivector-
Patch = ga.gatype.antibivector()                                             # Antibivector-
EvenMap = ga.gatype((Even, Point))
RateMap = ga.gatype((Bivector, Point))
StretchMap = ga.gatype((Scalar, Point))
triangle = Topology(3, [[0, 1, 2]])


def triangle_darboux(
    corners: Point, points: Point, corner_rotors: Rotor,
) -> tuple[Even, Rotor, EvenMap, RateMap, StretchMap, RateMap, Plane]:
    """Linearly blended corner rotors (`[3] Rotor+`) across sample points (`[...] Point-`) decompose their Darboux derivative into antisymmetric edge Whitney 1-forms in grade two and symmetric Regge differentials in grade zero."""
    sides = (triangle.boundary[2, 1] * (triangle.d[0] * corners.field())).batch()[:, 0]  # [3] Point-
    patch = (sides[0] & sides[1]) / 2                                        # [] Patch-
    gradients = (sides | patch.inverse()) / 2                                # [3] Plane+
    corner_weights = 1 + (gradients & (points[:, None] - corners))           # [..., 3] Scalar+

    # Linear blending cuts inside the unit rotor sphere; normalization restores a rigid turn.
    chord_rotors = (corner_rotors * corner_weights).sum(axis=-1)             # [...] Even+
    unit_rotors = chord_rotors.normalized()                                  # [...] Rotor+
    rotor_differential = (corner_rotors * (gradients & Point)).sum(axis=0)   # [] Even+ <- Point-
    darboux = 2 * chord_rotors.reverse() * rotor_differential                # [...] Even+ <- Point-

    # Reversing an edge flips its transition's turning plane and preserves its scalar sagitta.
    tails, heads = triangle.corners[2, 1]
    transitions = corner_rotors[tails].reverse() * corner_rotors[heads]      # [3] Rotor+
    edge_bivectors = 2 * transitions.select[2]                               # [3] Bivector+
    edge_sagittas = 2 * (transitions.select[0] - 1)                          # [3] Scalar+
    whitney_forms = corner_weights[:, tails] * gradients[heads] - corner_weights[:, heads] * gradients[tails]    # [..., 3] Plane+
    symmetric_forms = corner_weights[:, tails] * gradients[heads] + corner_weights[:, heads] * gradients[tails]  # [..., 3] Plane+
    whitney_rates = (edge_bivectors * (whitney_forms & Point)).sum(axis=-1)  # [...] Bivector+ <- Point-
    chord_stretch = (edge_sagittas * (symmetric_forms & Point)).sum(axis=-1) # [...] Scalar+ <- Point-
    angular_velocity = whitney_rates / chord_rotors.scalar_norm_squared()    # [...] Bivector+ <- Point-
    return chord_rotors, unit_rotors, darboux, whitney_rates, chord_stretch, angular_velocity, whitney_forms


def skyrmion_winding(
    topology: Topology, vertices: Point, twist_angle: float,
) -> tuple[Rotor, Point, Scalar, Scalar, Scalar, Scalar]:
    """A radial rotor texture (`Rotor+[V]`) wraps the director sphere across a triangulated disk while its face Darboux rates give the frame Dirichlet energy density (`Scalar+[F]`) and swept solid angles (`Scalar+[F]`)."""
    vertical = mv.z.dual()                                                   # [] Point-
    # Radial position and vertical direction span the turning plane; a half turn sends the rim to the south pole.
    rotors = ((-twist_angle / 2) * vertices.commutator(vertical)).exp()      # Rotor+[V]
    directors = rotors >> vertical                                           # Point-[V]
    scalar_parts = rotors.select[0]                                          # Scalar+[V]
    bivector_norms = rotors.select[2].norm()                                 # Scalar+[V]

    edges = topology.d[0] * vertices                                         # Point-[E]
    sides = topology.boundary[2, 1] * edges                                  # [3] Point-[F]
    patches = (sides[0] & sides[1]) / 2                                      # Patch-[F]
    areas = patches.norm()                                                   # Scalar+[F]
    gradients = (sides | patches.inverse()) / 2                              # [3] Plane+[F]

    face_rotors = topology.incidence[2, 0] * rotors                          # [3] Rotor+[F]
    face_chords = face_rotors.sum(axis=0) / 3                                # Even+[F]
    face_differential = (face_rotors * (gradients & Point)).sum(axis=0)      # (Even+ <- Point-)[F]
    face_darboux = 2 * face_chords.reverse() * face_differential             # (Even+ <- Point-)[F]
    face_rates = face_darboux.select[2] / face_chords.scalar_norm_squared()  # (Bivector+ <- Point-)[F]

    # Cotangent weights assemble edge readings into the direction-independent frame Dirichlet energy density.
    tails, heads = topology.corners[2, 1]
    dots = -sides[tails].dot(sides[heads])                                   # [3] Scalar+[F]
    cotangents = dots / (2 * areas)                                          # [3] Scalar+[F]
    energy_density = ((cotangents / (2 * areas)) * face_rates(sides).scalar_norm_squared()).sum(axis=0)  # Scalar+[F]

    # Commuting the two edge turning rates and pairing with the equatorial patch gives the director's swept area.
    swept_turns = face_rates(sides[1]).commutator(face_rates(sides[0])) / 2  # Bivector+[F]
    solid_angles = swept_turns & mv.xy.dual()                                # Scalar+[F]
    return rotors, directors, scalar_parts, bivector_norms, energy_density, solid_angles


def rotor_laplacian_operators(
    topology: Topology, vertices: Point,
) -> tuple[SparseExtensor, SparseExtensor, Scalar, Scalar, Point]:
    """Mixed-grade rotor coboundary (`(Even+ <- Even+)[E <- V]`), cotangent Laplacian (`(Even+ <- Even+)[V <- V]`), edge cotangent weights (`Scalar+[E]`), triangle areas (`Scalar+[F]`), and oriented triangle sides (`[3] Point-[F]`) follow from the surface geometry."""
    edges = topology.d[0] * vertices                                         # Point-[E]
    sides = topology.boundary[2, 1] * edges                                  # [3] Point-[F]
    patches = (sides[0] & sides[1]) / 2                                      # Patch-[F]
    areas = patches.norm()                                                   # Scalar+[F]
    tails, heads = topology.corners[2, 1]
    dots = -sides[tails].dot(sides[heads])                                   # [3] Scalar+[F]
    cotangents = dots / (2 * areas)                                          # [3] Scalar+[F]
    edge_weights = (topology.incidence[2, 1].adjugate() * (cotangents / 2)).sum(axis=0)  # Scalar+[E]
    rotor_d0 = topology.d[0] * Rotor                                         # (Even+ <- Even+)[E <- V]
    rotor_laplacian = rotor_d0.adjoint()(spdiag(edge_weights) * rotor_d0)    # (Even+ <- Even+)[V <- V]
    return rotor_d0, rotor_laplacian, edge_weights, areas, sides


def harmonic_rotor_frames(
    topology: Topology, vertices: Point, boundary_indices: np.ndarray, prescribed_rotors: Rotor,
) -> tuple[Even, Rotor]:
    """Boundary rotors (`[...] Rotor+[V]`) on `boundary_indices` extend across interior vertices into a harmonic chord field (`[...] Even+[V]`) and its unit-normalized rotor field (`[...] Rotor+[V]`)."""
    _, rotor_laplacian, _, _, _ = rotor_laplacian_operators(topology, vertices)
    free_vertices = topology.interior[0]
    free_inclusion = topology.reading(free_vertices, 0).adjugate() * Rotor   # (Even+ <- Even+)[V <- V_free]
    boundary_inclusion = topology.reading(boundary_indices, 0).adjugate() * Rotor  # (Even+ <- Even+)[V <- V_boundary]

    # Solving in the 4D even subalgebra minimizes chord Dirichlet energy; normalization restores unit rotors.
    boundary_values = boundary_inclusion(boundary_inclusion.adjoint()(prescribed_rotors))  # [...] Even+[V]
    rhs = -free_inclusion.adjoint()(rotor_laplacian(boundary_values))        # [...] Even+[V_free]
    harmonic_chords = boundary_values + free_inclusion(
        free_inclusion.adjoint()(rotor_laplacian(free_inclusion)).solve(rhs)
    )                                                                        # [...] Even+[V]
    harmonic_rotors = harmonic_chords.normalized()                           # [...] Rotor+[V]
    return harmonic_chords, harmonic_rotors


def rotor_relaxation(
    topology: Topology, vertices: Point, initial_rotors: Rotor,
    step_size: float, n_steps: int,
) -> Iterator[Rotor]:
    """Yields the unit rotor field (`Rotor+[V]`) over `n_steps + 1` projected cotangent smoothing steps with fixed boundary frames."""
    _, rotor_laplacian, _, _, _ = rotor_laplacian_operators(topology, vertices)
    free_vertices = topology.interior[0]
    free_inclusion = topology.reading(free_vertices, 0).adjugate() * Rotor   # (Even+ <- Even+)[V <- V_free]
    current_rotors = initial_rotors                                          # Rotor+[V]
    for _ in range(n_steps + 1):
        yield current_rotors
        current_rotors = (
            current_rotors - step_size * free_inclusion(free_inclusion.adjoint()(rotor_laplacian(current_rotors)))
        ).normalized()                                                       # Rotor+[V]
