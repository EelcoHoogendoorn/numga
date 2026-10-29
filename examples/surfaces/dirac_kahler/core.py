"""A geometric derivative joins scalar and stream potentials into one field of waves.

The product `corner_gradients * Full` differentiates a vertex multivector by grade. Its restriction to `Even` pairs source-to-sink gradients with circulating stream gradients; the adjugate returns their face field to the vertices. Separately, the Whitney maps reconstruct cochains on vertices, edges and faces into a mixed-grade field at face centres. A velocity, an intrinsic antivector, splits into a potential part that the medium drives, `isotropic_medium(gradient(potential))`, a stream part that needs no metric, `gradient(stream) ^ patches`, and a harmonic part circulating around a hole.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga.algebras import VGA2D as ga
from numga.sparse import SparseExtensor, spdiag
from examples import instantiate

mesh_module = instantiate("examples.util.mesh", ga)
Mesh, as_scalar, mv = mesh_module.Mesh, mesh_module.as_scalar, mesh_module.mv
Scalar = ga.gatype.scalar()                                                  # Scalar+
Plane = ga.gatype.vector()                                                   # Vector+
Bivector = ga.gatype.bivector()                                              # Bivector+
Even = ga.gatype.even()                                                      # Even+
Full = ga.gatype.full()                                                      # Full+
Point = ga.gatype.antivector()                                               # Antivector-


# --- math -----------------------------------------------------------------------------
class Operators:
    """An oriented triangle mesh carries the Whitney reconstruction maps and the mixed-grade Dirac–Kähler operators."""

    def __init__(self, mesh: Mesh, vertices: Point):
        self.mesh = mesh
        self.d1 = mesh.d[1] * Scalar                                         # (Scalar+ <- Scalar+)[faces <- edges]
        edges = mesh.d[0] * vertices                                         # Point-[edges]
        self.sides = mesh.boundary[2, 1] * edges                             # [corners] Point-[faces]
        self.patches = (self.sides[0] & self.sides[1]) / 2                   # Patch-[faces]
        self.areas = self.patches.norm()                                     # Scalar+[faces]

        # Dividing each opposite edge by twice the signed triangle patch gives the three constant corner gradients.
        self.corner_gradients = self.sides / (2 * self.patches)              # [corners] Plane+[faces]
        thirds = mesh.incidence[2, 0].mean(axis=0)                           # Scalar+[faces <- vertices]
        self.whitney0 = thirds * Scalar                                      # (Scalar+ <- Scalar+)[faces <- vertices]
        self.gradient = (spdiag(self.corner_gradients) * mesh.incidence[2, 0]).sum(axis=0) * Scalar  # (Plane+ <- Scalar+)[faces <- vertices]

        # At each face centroid the three barycentric hats equal 1/3, so each edge 1-form
        # reduces to one third of the difference of its endpoints' corner gradients.
        centroid_forms = (self.corner_gradients[[2, 0, 1]] - self.corner_gradients[[1, 2, 0]]) / 3  # [corners] Plane+[faces]
        self.whitney1 = (spdiag(centroid_forms) * mesh.boundary[2, 1]).sum(axis=0) * Scalar  # (Plane+ <- Scalar+)[faces <- edges]
        first, second = self.corner_gradients[[1, 2, 0]], self.corner_gradients[[2, 0, 1]]  # [corners] Plane+[faces], [corners] Plane+[faces]
        self.curl = (spdiag(2 * (first ^ second)) * mesh.boundary[2, 1]).sum(axis=0) * Scalar  # (Bivector+ <- Scalar+)[faces <- edges]
        self.whitney2 = spdiag(mv.xy / (mv.xy & self.patches) * Scalar)      # (Bivector+ <- Scalar+)[faces <- faces]

        # Multiplying the corner gradients by a full multivector unites divergence in grade 0,
        # gradient plus rotated bivector gradient in grade 1, and exterior curl in grade 2.
        self.dirac = (spdiag(self.corner_gradients) * mesh.incidence[2, 0]).sum(axis=0) * Full  # (Full+ <- Full+)[faces <- vertices]

        # An isotropic medium carries each face's gradient to the area-weighted velocity it drives, through the metric;
        # its inverse reads a velocity as the gradient that drives it.
        self.isotropic_medium = spdiag(self.areas * Plane.dual())            # (Point- <- Plane+)[faces <- faces]
        self.isotropic_resistance = spdiag(Plane.dual().inverse() / self.areas)  # (Plane+ <- Point-)[faces <- faces]

        # Restricting the right factor to the even subalgebra maps scalar potentials and stream bivectors
        # together into a pure grade-1 face 1-form.
        self.even_gradient = (spdiag(self.corner_gradients) * mesh.incidence[2, 0]).sum(axis=0) * Even  # (Plane+ <- Even+)[faces <- vertices]
        self.vertex_areas = ~thirds * self.areas                             # Scalar+[vertices]
        inverse_vertex_mass = spdiag((1 / self.vertex_areas) * Even.dual().inverse())  # (Even+ <- Even-)[vertices <- vertices]
        # Because the 2D regressive pairing flips sign on odd grades and keeps it on even grades,
        # the adjugate of the even gradient is already the skew-adjoint partner that conserves wave energy.
        self.even_divergence = inverse_vertex_mass(self.even_gradient.adjugate()(self.isotropic_medium))  # (Even+ <- Plane+)[vertices <- faces]

        # A velocity is built from a potential through the medium, and from a stream function without a metric:
        # meeting its gradient with the patch gives the area-weighted velocity along its level curves.
        # Applying the medium to the even gradient produces both parts at once on `potential - stream * mv.xy`.
        self.potential_velocity = self.isotropic_medium(self.gradient)       # (Point- <- Scalar+)[faces <- vertices]
        self.stream_velocity = spdiag(Plane ^ self.patches)(self.gradient)   # (Point- <- Scalar+)[faces <- vertices]
        self.even_velocity = self.isotropic_medium(self.even_gradient)       # (Point- <- Even+)[faces <- vertices]
        self.potential_laplacian = self.gradient.adjugate()(self.potential_velocity)  # (Antiscalar- <- Scalar+)[vertices <- vertices]
        self.stream_laplacian = self.stream_velocity.adjugate()(self.isotropic_resistance(self.stream_velocity))  # (Antiscalar- <- Scalar+)[vertices <- vertices]
        self.even_laplacian = self.even_gradient.adjugate()(self.even_velocity)  # (Even- <- Even+)[vertices <- vertices]

def harmonic_circulation(operators: Operators, inner_ring: np.ndarray,
                         interior: np.ndarray) -> tuple[Scalar, Point]:
    """The harmonic stream function is one on the inner rim and zero on the outer, and its velocity circulates around the hole."""
    mesh = operators.mesh
    pinned = SparseExtensor.selection(mv.context, inner_ring, mesh.n_vertices)
    boundary_stream = ~pinned * as_scalar(np.ones(len(inner_ring))).field()  # Scalar+[vertices]
    free = ~SparseExtensor.selection(mv.context, interior, mesh.n_vertices) * Scalar  # (Scalar+ <- Scalar+)[vertices <- interior]
    rhs = -free.adjugate()(operators.stream_laplacian(boundary_stream))      # Antiscalar-[interior]
    stream = boundary_stream + free(free.adjugate()(operators.stream_laplacian(free)).solve(rhs))  # Scalar+[vertices]
    return stream, operators.stream_velocity(stream)


def hodge_decomposition(operators: Operators, flow: Point,
                        interior: np.ndarray) -> tuple[Scalar, Scalar, Point, Point, Point]:
    """One mixed-grade `Even` Poisson solve on the interior recovers both the scalar potential and the stream bivector that split `flow` into its orthogonal potential, stream and harmonic parts."""
    mesh = operators.mesh
    free = ~SparseExtensor.selection(mv.context, interior, mesh.n_vertices) * Even  # (Even+ <- Even+)[vertices <- interior]
    # On every interior vertex star the adjugate of the even gradient reads the flow's divergence in grade 0
    # and its circulation in grade 2, so one solve on the even subalgebra recovers both potentials together.
    readings = free.adjugate()(operators.even_gradient.adjugate()(flow))     # Even-[interior]
    even_potential = free(free.adjugate()(operators.even_laplacian(free)).solve(readings))  # Even+[vertices]
    potential = even_potential.select[0]                                     # Scalar+[vertices]
    stream = -(even_potential.select[2] & mv.xy.dual())                      # Scalar+[vertices]
    potential_flow = operators.even_velocity(potential)                      # Point-[faces]
    stream_flow = operators.even_velocity(even_potential.select[2])          # Point-[faces]
    harmonic_flow = flow - potential_flow - stream_flow                      # Point-[faces]
    return potential, stream, potential_flow, stream_flow, harmonic_flow


def propagate(operators: Operators, even_state: Even, face_field: Plane,
              dt: float, n_steps: int) -> Iterator[tuple[Even, Plane]]:
    """Yields the fourth-order Runge–Kutta trajectory of an even potential and a face 1-form under the Dirac–Kähler wave operators."""
    div, grad = operators.even_divergence, operators.even_gradient
    yield even_state, face_field
    for _ in range(n_steps):
        k1_even, k1_field = div(face_field), grad(even_state)                    # Even+[vertices], Plane+[faces]
        k2_even, k2_field = div(face_field + 0.5 * dt * k1_field), grad(even_state + 0.5 * dt * k1_even)  # Even+[vertices], Plane+[faces]
        k3_even, k3_field = div(face_field + 0.5 * dt * k2_field), grad(even_state + 0.5 * dt * k2_even)  # Even+[vertices], Plane+[faces]
        k4_even, k4_field = div(face_field + dt * k3_field), grad(even_state + dt * k3_even)  # Even+[vertices], Plane+[faces]
        even_state = even_state + (dt / 6) * (k1_even + 2 * k2_even + 2 * k3_even + k4_even)  # Even+[vertices]
        face_field = face_field + (dt / 6) * (k1_field + 2 * k2_field + 2 * k3_field + k4_field)  # Plane+[faces]
        yield even_state, face_field
