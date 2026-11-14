"""Mixed-grade Dirac–Kähler derivatives, Whitney cochain maps, Hodge decomposition, and boundary-coupled waves.

Multiplying a triangle's simplex gradients by `Full` differentiates a vertex multivector across grades in one
product. Restricting that product to `Even` combines source-to-sink potential slopes with circulating stream
slopes in one face 1-form, while its adjugate through the medium returns the divergence and circulation to the
vertices. Separately, the Whitney maps reconstruct cochains on vertices, edges, and faces into a mixed-grade
field at face centroids. On a perforated annulus, an intrinsic velocity field splits into a metric-driven
potential flow, a metric-free stream flow along level curves, and a harmonic circulation around the hole.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import NumpyContext
from numga.algebras import VGA2D as ga
from numga.extensor import Extensor
from numga.sparse import spdiag
from numga.topology import Topology

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                                                  # Scalar+
Plane = ga.gatype.vector()                                                   # Vector+
Bivector = ga.gatype.bivector()                                              # Bivector+
Even = ga.gatype.even()                                                      # Even+
Full = ga.gatype.full()                                                      # Full+
Point = ga.gatype.antivector()                                               # Antivector-


# --- math -----------------------------------------------------------------------------
def simplex_geometry(topology: Topology, vertices: Point) -> tuple[Extensor, Extensor, Scalar, Plane]:
    """Opposite triangle edges and signed patches determine the face areas and corner simplex gradients (`[3] Plane+[F]`)."""
    edges = topology.d[0] * vertices                                         # Point-[E]
    sides = topology.boundary[2, 1] * edges                                  # [3] Point-[F]
    patches = (sides[0] & sides[1]) / 2                                      # Antibivector-[F]
    areas = patches.norm()                                                   # Scalar+[F]
    # Dividing each opposite edge by twice the signed triangle patch gives the three constant simplex gradients.
    gradients = sides / (2 * patches)                                        # [3] Plane+[F]
    return sides, patches, areas, gradients


def whitney_maps(
    topology: Topology, patches: Extensor, gradients: Plane,
) -> tuple[Extensor, Extensor, Extensor, Extensor]:
    """Corner means and simplex-gradient differences embed 0-, 1-, and 2-cochains into face fields alongside the exterior coboundary."""
    exterior_curl = topology.d[1] * Scalar                                   # (Scalar+ <- Scalar+)[F <- E]
    corner_mean = topology.incidence[2, 0].mean(axis=0)                      # Scalar+[F <- V]
    whitney0 = corner_mean * Scalar                                          # (Scalar+ <- Scalar+)[F <- V]
    # At each triangle centroid the three corner weights equal 1/3, so each edge 1-form is one third of its endpoints' simplex-gradient difference.
    tails, heads = topology.corners[2, 1]
    centroid_forms = (gradients[heads] - gradients[tails]) / 3               # [3] Plane+[F]
    whitney1 = (spdiag(centroid_forms) * topology.oriented[2, 1]).sum(axis=0) * Scalar  # (Plane+ <- Scalar+)[F <- E]
    whitney2 = mv.xy / (mv.xy & patches) * Scalar                            # (Bivector+ <- Scalar+)[F]
    return exterior_curl, whitney0, whitney1, whitney2


def dirac_derivatives(topology: Topology, gradients: Plane) -> tuple[Extensor, Extensor, Extensor]:
    """Summing the simplex gradients over each triangle's corners yields the scalar gradient, the mixed-grade Dirac–Kähler derivative, and its `Even` restriction."""
    corners = topology.incidence[2, 0]                                       # [3] Scalar+[F <- V]
    gradient = (spdiag(gradients) * corners).sum(axis=0) * Scalar            # (Plane+ <- Scalar+)[F <- V]
    # Multiplying the simplex gradients by a full multivector unites divergence in grade 0, gradient plus stream slope in grade 1, and curl in grade 2.
    dirac = (spdiag(gradients) * corners).sum(axis=0) * Full                 # (Full+ <- Full+)[F <- V]
    even_gradient = (spdiag(gradients) * corners).sum(axis=0) * Even         # (Plane+ <- Even+)[F <- V]
    return gradient, dirac, even_gradient


def medium_operators(
    topology: Topology, patches: Extensor, areas: Scalar, gradient: Extensor, even_gradient: Extensor,
) -> tuple[Extensor, Extensor, Extensor, Extensor, Extensor, Extensor, Extensor]:
    """The isotropic metric medium and metric-free patch wedge build the potential, stream, and `Even` velocity maps and their Laplacians."""
    # An isotropic medium carries each face's slope through the metric to the area-weighted velocity it drives; its inverse reads back the driving slope.
    isotropic_medium = spdiag(areas * Plane.dual())                          # (Point- <- Plane+)[F <- F]
    isotropic_resistance = Plane.dual().inverse() / areas                    # (Plane+ <- Point-)[F]
    potential_velocity = isotropic_medium(gradient)                          # (Point- <- Scalar+)[F <- V]
    # Meeting a stream slope with the triangle patch gives the area-weighted velocity along level curves without a metric.
    stream_velocity = (Plane ^ patches)(gradient)                            # (Point- <- Scalar+)[F <- V]
    even_velocity = isotropic_medium(even_gradient)                          # (Point- <- Even+)[F <- V]
    stream_laplacian = stream_velocity.adjugate()(isotropic_resistance(stream_velocity))  # (Antiscalar- <- Scalar+)[V <- V]
    even_laplacian = even_gradient.adjugate()(even_velocity)                 # (Even- <- Even+)[V <- V]
    return (
        isotropic_medium, isotropic_resistance, potential_velocity,
        stream_velocity, even_velocity, stream_laplacian, even_laplacian,
    )


def harmonic_circulation(
    topology: Topology, stream_velocity: Extensor, stream_laplacian: Extensor,
    inner_ring: np.ndarray, interior: np.ndarray,
) -> tuple[Scalar, Point]:
    """Pinning the stream function to one on the inner rim and zero on the outer rim yields the harmonic circulation velocity (`Point-[F]`)."""
    pinned = topology.reading(inner_ring, 0)                                 # Scalar+[V_inner <- V]
    boundary_stream = pinned.adjugate() * mv.scalar(np.ones(len(inner_ring))[..., None]).field()  # Scalar+[V]
    free_scalar = topology.reading(interior, 0).adjugate() * Scalar          # (Scalar+ <- Scalar+)[V <- V_interior]
    boundary_load = -free_scalar.adjugate()(stream_laplacian(boundary_stream))  # Antiscalar-[V_interior]
    interior_laplacian = free_scalar.adjugate()(stream_laplacian(free_scalar))  # (Antiscalar- <- Scalar+)[V_interior <- V_interior]
    harmonic_stream = boundary_stream + free_scalar(interior_laplacian.solve(boundary_load))  # Scalar+[V]
    return harmonic_stream, stream_velocity(harmonic_stream)


def hodge_decomposition(
    topology: Topology, even_gradient: Extensor, even_velocity: Extensor, even_laplacian: Extensor,
    total_flow: Point, interior: np.ndarray,
) -> tuple[Scalar, Scalar, Point, Point, Point]:
    """One interior `Even` Poisson solve recovers the scalar potential and stream function that split `total_flow` into potential, stream, and harmonic flows (`Point-[F]`)."""
    free_even = topology.reading(interior, 0).adjugate() * Even              # (Even+ <- Even+)[V <- V_interior]
    # On every interior vertex star, the adjugate of the even gradient reads divergence in grade 0 and circulation in grade 2 together.
    readings = free_even.adjugate()(even_gradient.adjugate()(total_flow))    # Even-[V_interior]
    interior_laplacian = free_even.adjugate()(even_laplacian(free_even))     # (Even- <- Even+)[V_interior <- V_interior]
    even_potential = free_even(interior_laplacian.solve(readings))           # Even+[V]
    potential = even_potential.select[0]                                     # Scalar+[V]
    stream = -(mv.xy.dual() & even_potential.select[2])                      # Scalar+[V]
    potential_flow = even_velocity(potential)                                # Point-[F]
    stream_flow = even_velocity(even_potential.select[2])                    # Point-[F]
    harmonic_flow = total_flow - potential_flow - stream_flow                # Point-[F]
    return potential, stream, potential_flow, stream_flow, harmonic_flow


def wave_divergence(
    topology: Topology, areas: Scalar, even_gradient: Extensor, isotropic_medium: Extensor,
) -> tuple[Scalar, Extensor]:
    """Dual-cell vertex areas and the negative adjugate of `even_gradient` through `isotropic_medium` form the skew-adjoint `Even` divergence."""
    dual_areas = topology.incidence[2, 0].mean(axis=0).adjugate() * areas    # Scalar+[V]
    inverse_vertex_mass = (1 / dual_areas) * Even.dual().inverse()           # (Even+ <- Even-)[V]
    # Minus the even gradient's adjugate through the medium makes divergence and gradient skew-adjoint partners in wave energy.
    even_divergence = -inverse_vertex_mass(even_gradient.adjugate()(isotropic_medium))  # (Even+ <- Plane+)[V <- F]
    return dual_areas, even_divergence


def wave_states(
    even_gradient: Extensor, even_divergence: Extensor, even_state: Even, face_field: Plane,
    dt: float, n_steps: int, stride: int,
) -> Iterator[tuple[Even, Plane]]:
    """Yields every `stride`-th fourth-order Runge–Kutta state (`Even+[V]`, `Plane+[F]`) of the coupled Dirac–Kähler wave."""
    yield even_state, face_field
    for step in range(1, n_steps + 1):
        first_even, first_field = even_divergence(face_field), even_gradient(even_state)  # Even+[V], Plane+[F]
        second_even = even_divergence(face_field + 0.5 * dt * first_field)   # Even+[V]
        second_field = even_gradient(even_state + 0.5 * dt * first_even)     # Plane+[F]
        third_even = even_divergence(face_field + 0.5 * dt * second_field)   # Even+[V]
        third_field = even_gradient(even_state + 0.5 * dt * second_even)     # Plane+[F]
        fourth_even = even_divergence(face_field + dt * third_field)         # Even+[V]
        fourth_field = even_gradient(even_state + dt * third_even)           # Plane+[F]
        even_state = even_state + (dt / 6) * (first_even + 2 * second_even + 2 * third_even + fourth_even)  # Even+[V]
        face_field = face_field + (dt / 6) * (first_field + 2 * second_field + 2 * third_field + fourth_field)  # Plane+[F]
        if step % stride == 0:
            yield even_state, face_field
