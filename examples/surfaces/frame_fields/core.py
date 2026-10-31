"""Rotations attached to a mesh become a continuous field of orientations. Blending corner rotors traces a chord inside the unit rotor sphere; its changing orientation and changing length occupy different grades of one derivative.

The Darboux map `2 * chord_rotors.reverse() * rotor_differential` takes a displacement to that mixed-grade change. Its turning plane carries Whitney's edge forms. This example follows the local construction into a disk that wraps the director sphere, and boundary frames extended across a curved surface.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga.algebras import VGA3D as ga
from numga.sparse import SparseExtensor, spdiag
from examples import instantiate

mesh_module = instantiate("examples.util.mesh", ga)
Mesh, as_scalar, mv = mesh_module.Mesh, mesh_module.as_scalar, mesh_module.mv
Scalar = ga.gatype.scalar()                                                  # Scalar+
Plane = ga.gatype.vector()                                                   # Vector+
Bivector = ga.gatype.bivector()                                              # Bivector+
Even = ga.gatype.even()                                                      # Even+
Rotor = ga.gatype.rotor()                                                    # Rotor+
Point = ga.gatype.antivector()                                               # Antivector-
EvenMap = ga.gatype((Even, Point))
RateMap = ga.gatype((Bivector, Point))
StretchMap = ga.gatype((Scalar, Point))
triangle = Mesh(3, [[0, 1, 2]])


# --- math -----------------------------------------------------------------------------
def triangle_darboux(
    corners: Point, points: Point, corner_rotors: Rotor,
) -> tuple[Even, Rotor, EvenMap, RateMap, StretchMap, RateMap, Plane]:
    """Linearly blended corner rotors on a triangle decompose their Darboux derivative into edge Whitney 1-forms in grade two and symmetric Regge differentials in grade zero."""
    sides = (triangle.boundary[2, 1] * (triangle.d[0] * corners.field())).batch()[:, 0]  # [corners] Point-
    patch = (sides[0] & sides[1]) / 2                                        # [] Patch-
    gradients = (sides | patch.inverse()) / 2                                # [corners] Plane+
    hats = 1 + (gradients & (points[:, None] - corners))                     # [samples, corners] Scalar+

    # Linear blending cuts inside the unit rotor sphere; normalization restores a rigid turn.
    chord_rotors = (corner_rotors * hats).sum(axis=-1)                       # [samples] Even+
    unit_rotors = chord_rotors.normalized()                                  # [samples] Rotor+
    rotor_differential = (corner_rotors * (gradients & Point)).sum(axis=0)   # [] Even+ <- Point-
    darboux = 2 * chord_rotors.reverse() * rotor_differential                # [samples] Even+ <- Point-

    # Reversing an edge flips its transition's turning plane and preserves its scalar part.
    # These match the antisymmetric Whitney forms and symmetric endpoint-product differentials.
    tails, heads = triangle.corners[2, 1]
    transitions = corner_rotors[tails].reverse() * corner_rotors[heads]      # [corners] Rotor+
    edge_bivectors = 2 * transitions.select[2]                               # [corners] Bivector+
    edge_sagittas = 2 * (transitions.select[0] - 1)                          # [corners] Scalar+
    whitney_forms = hats[:, tails] * gradients[heads] - hats[:, heads] * gradients[tails]            # [samples, corners] Plane+
    symmetric_forms = hats[:, tails] * gradients[heads] + hats[:, heads] * gradients[tails]          # [samples, corners] Plane+
    whitney_rates = (edge_bivectors * (whitney_forms & Point)).sum(axis=-1)  # [samples] Bivector+ <- Point-
    chord_stretch = (edge_sagittas * (symmetric_forms & Point)).sum(axis=-1) # [samples] Scalar+ <- Point-
    angular_velocity = whitney_rates / chord_rotors.scalar_norm_squared()    # [samples] Bivector+ <- Point-
    return chord_rotors, unit_rotors, darboux, whitney_rates, chord_stretch, angular_velocity, whitney_forms


def skyrmion_winding(
    mesh: Mesh, vertices: Point, twist_angle: float,
) -> tuple[Rotor, Point, Scalar, Scalar, Scalar, Scalar]:
    """A radial rotor texture wraps the sphere of directors; its frame derivative reads the energy and swept-area density on each face."""
    vertical = mv.z.dual()                                                   # [] Point-
    # The radial position and vertical direction span the plane of turning.
    # At a half turn across the unit disk, its director runs from the north to the south pole.
    rotors = ((-twist_angle / 2) * vertices.commutator(vertical)).exp()      # Rotor+[vertices]
    directors = rotors >> vertical                                           # Point-[vertices]
    scalar_parts = rotors.select[0]                                          # Scalar+[vertices]
    bivector_norms = rotors.select[2].norm()                                 # Scalar+[vertices]

    edges = mesh.d[0] * vertices                                             # Point-[edges]
    sides = mesh.boundary[2, 1] * edges                                      # [corners] Point-[faces]
    patches = (sides[0] & sides[1]) / 2                                      # Patch-[faces]
    areas = patches.norm()                                                   # Scalar+[faces]
    corner_gradients = (sides | patches.inverse()) / 2                       # [corners] Plane+[faces]

    face_rotors = mesh.incidence[2, 0] * rotors                              # [corners] Rotor+[faces]
    chord_rotors = face_rotors.sum(axis=0) / 3                               # Even+[faces]
    face_differential = (face_rotors * (corner_gradients & Point)).sum(axis=0)  # (Even+ <- Point-)[faces]
    face_darboux = 2 * chord_rotors.reverse() * face_differential            # (Even+ <- Point-)[faces]
    face_rates = face_darboux.select[2] / chord_rotors.scalar_norm_squared() # (Bivector+ <- Point-)[faces]

    # The cotangent weights turn edge readings into the direction-independent frame energy.
    tails, heads = mesh.corners[2, 1]
    dots = -sides[tails].dot(sides[heads])                                   # [corners] Scalar+[faces]
    cotangents = dots / (2 * areas)                                          # [corners] Scalar+[faces]
    energy_density = ((cotangents / (2 * areas)) * face_rates(sides).scalar_norm_squared()).sum(axis=0)  # Scalar+[faces]

    # Reading the two turning rates against the director's equatorial patch gives its swept area.
    # The full frame connection is flat: its derivative cancels this commutator term.
    swept_turns = face_rates(sides[1]).commutator(face_rates(sides[0])) / 2    # Bivector+[faces]
    solid_angles = swept_turns & mv.xy.dual()                                # Scalar+[faces]
    return rotors, directors, scalar_parts, bivector_norms, energy_density, solid_angles


def rotor_laplacian_operators(
    mesh: Mesh, vertices: Point,
) -> tuple[SparseExtensor, SparseExtensor, Scalar, Scalar, Point]:
    """The mixed-grade Rotor coboundary, cotangent Laplacian, edge cotangent weights, face areas, and face sides on a surface mesh."""
    edges = mesh.d[0] * vertices                                             # Point-[edges]
    sides = mesh.boundary[2, 1] * edges                                      # [corners] Point-[faces]
    patches = (sides[0] & sides[1]) / 2                                      # Patch-[faces]
    areas = patches.norm()                                                   # Scalar+[faces]
    tails, heads = mesh.corners[2, 1]
    dots = -sides[tails].dot(sides[heads])                                   # [corners] Scalar+[faces]
    cotangents = dots / (2 * areas)                                          # [corners] Scalar+[faces]
    edge_weights = (~mesh.incidence[2, 1] * (cotangents / 2)).sum(axis=0)    # Scalar+[edges]
    rotor_d0 = mesh.d[0] * Rotor                                             # (Even+ <- Even+)[edges <- vertices]
    rotor_laplacian = rotor_d0.adjoint()(spdiag(edge_weights) * rotor_d0)    # (Even+ <- Even+)[vertices <- vertices]
    return rotor_d0, rotor_laplacian, edge_weights, areas, sides


def harmonic_rotor_frames(
    mesh: Mesh, vertices: Point, boundary: np.ndarray, prescribed_rotors: Rotor,
) -> tuple[Even, Rotor]:
    """Pinned boundary rotors determine a linear harmonic chord field in the 4D even subalgebra, and its normalized unit rotor field."""
    _, rotor_laplacian, _, _, _ = rotor_laplacian_operators(mesh, vertices)
    free_vertices = mesh.interior(0)
    free = ~mesh.reading(free_vertices) * Rotor                              # (Even+ <- Even+)[vertices <- free]
    pinned = ~mesh.reading(boundary) * Rotor                                 # (Even+ <- Even+)[vertices <- boundary]

    # The chord field minimizes edge energy in the even subalgebra.
    # Normalization gives unit frames, changing the variational problem in the process.
    boundary_values = pinned(pinned.adjoint()(prescribed_rotors))            # [...] Even+[vertices]
    rhs = -free.adjoint()(rotor_laplacian(boundary_values))                  # [...] Even+[free]
    harmonic_chords = boundary_values + free(free.adjoint()(rotor_laplacian(free)).solve(rhs))  # [...] Even+[vertices]
    harmonic_rotors = harmonic_chords.normalized()                           # [...] Rotor+[vertices]
    return harmonic_chords, harmonic_rotors


def relax_rotor_frames(
    mesh: Mesh, vertices: Point, initial_rotors: Rotor, boundary: np.ndarray,
    step_size: float, n_steps: int,
) -> Iterator[tuple[Rotor, Scalar]]:
    """Projected cotangent steps smooth a frame field with fixed boundary frames and unit rotors at every vertex."""
    rotor_d0, rotor_laplacian, edge_weights, _, _ = rotor_laplacian_operators(mesh, vertices)
    free_vertices = mesh.interior(0)
    free = ~mesh.reading(free_vertices) * Rotor                              # (Even+ <- Even+)[vertices <- free]
    current = initial_rotors                                                 # Rotor+[vertices]
    for _ in range(n_steps + 1):
        edge_differences = rotor_d0(current)                                 # Even+[edges]
        energy = (edge_weights * edge_differences.scalar_norm_squared()).sites.sum() / 2  # [] Scalar+
        yield current, energy
        current = (current - step_size * free(free.adjoint()(rotor_laplacian(current)))).normalized()  # Rotor+[vertices]
