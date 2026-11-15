"""Angle-weighted normal-spread flow creases a triangulated surface into developable patches.

A developable surface unrolls onto a plane without stretching or tearing because it bends in at most one direction at every point. Following Stein, Grinspun, and Crane (2018), the angle-weighted sum of face-normal dyads `normals.dual() * (normals & Point)` around each vertex measures how far the surrounding faces depart from sharing a common ruling direction; its smallest eigenvalue vanishes whenever one direction meets every incident face normal at a right angle.

Summing those smallest eigenvalues over the vertices defines the developability energy. Differentiating each vertex's minimal quadratic form at fixed minimizing direction combines two face contributions, one from turning the face normal `edges[0] & edges[1]` and one from opening the interior corner angles, and dividing each vertex gradient by the local normal-turning curvature scales steps uniformly across large and small triangles.
"""

from __future__ import annotations

from collections.abc import Iterator

import jax

from numga import Extensor, NumpyContext
from numga.algebras import VGA3D as ga
from numga.backend.jax import JaxContext
from numga.sparse import spdiag
from numga.topology import Topology

mv = NumpyContext(ga).multivector

Scalar = ga.gatype.scalar()                         # Scalar+
Plane = ga.gatype.vector()                          # Vector+
Point = ga.gatype.antivector()                      # Antivector-
Patch = ga.gatype.antibivector()                    # Antibivector-


# --- math -----------------------------------------------------------------------------
def corner_angles(topology: Topology, edges: Point) -> Scalar:
    """Interior angles (`[3] Scalar+[F]`) at the three corners of each triangle measure the turn between the two incident `edges` (`[3] Point-[F]`)."""
    tails, heads = topology.corners[2, 1]                                       # [3] Scalar+[F <- F], [3] Scalar+[F <- F]
    next_edges, previous_edges = edges[tails], edges[heads]                     # [3] Point-[F], [3] Point-[F]
    return (-(next_edges.dual_inverse() & previous_edges) / (next_edges.norm() * previous_edges.norm())).arccos()  # [3] Scalar+[F]


def normal_spread(topology: Topology, edges: Point, angles: Scalar) -> tuple[Patch, Extensor]:
    """Unit triangle patches (`Patch-[F]`) and angle-weighted normal-dyad maps (`(Point- <- Point-)[V]`) measure how far the faces around each vertex spread from a single great circle."""
    # Two edge displacements meet in the triangle's oriented normal patch.
    normals = (edges[0] & edges[1]).normalized()                                # Patch-[F]
    incident_angles = (topology.incidence[2, 0].adjugate() * spdiag(angles)).sum(axis=0)  # Scalar+[V <- F]
    return normals, incident_angles * (normals.dual() * (normals & Point))      # Patch-[F], (Point- <- Point-)[V]


def energy_gradient(topology: Topology, edges: Point, angles: Scalar, normals: Patch, directions: Point) -> Plane:
    """The vertex gradient (`Plane+[V]`) of the developability energy at fixed least-spread `directions` (`Point-[V]`) sums each triangle's normal-turning and corner-angle variations."""
    double_areas = (edges[0] & edges[1]).norm()                                 # Scalar+[F]
    corner_directions = topology.incidence[2, 0] * directions                   # [3] Point-[F]
    # Project each corner's minimizing direction onto the triangle's unit normal patch.
    normal_projections = normals & corner_directions                            # [3] Scalar-[F]
    weighted_directions = (corner_directions * (normal_projections * angles)).sum(axis=0)  # Bivector+[F]
    tangent_directions = 2 * (weighted_directions - normals.dual() * (weighted_directions & normals))  # Bivector+[F]
    # Join the tangential direction rate with the opposite edge over twice the triangle area.
    normal_variation = (tangent_directions & edges) / double_areas              # [3] Plane+[F]
    # Moving an edge's endpoints oppositely along its in-plane normal opens one corner angle and closes the other.
    tails, heads = topology.corners[2, 1]                                       # [3] Scalar+[F <- F], [3] Scalar+[F <- F]
    angle_variation = (edges | normals) / edges.scalar_norm_squared() * (normal_projections[heads].squared() - normal_projections[tails].squared())  # [3] Plane+[F]
    return (topology.incidence[2, 0].adjugate() * normal_variation + (topology.oriented[2, 1] * topology.d[0]).adjugate() * angle_variation).sum(axis=0)  # Plane+[V]


def vertex_stiffness(topology: Topology, edges: Point, angles: Scalar) -> Scalar:
    """Diagonal normal-turning curvature (`Scalar+[V]`) sums each incident corner's angle divided by its squared altitude above the opposite edge."""
    double_areas = (edges[0] & edges[1]).norm()                                 # Scalar+[F]
    return (topology.incidence[2, 0].adjugate() * (angles * edges.scalar_norm_squared() / double_areas.squared())).sum(axis=0)  # Scalar+[V]


def developability_flow(topology: Topology, vertices: Point, step_scale: float, n_states: int, n_stride: int) -> Iterator[tuple[Point, Scalar]]:
    """Yields `n_states` snapshots of vertex positions (`Point-[V]`) and minimal spread eigenvalues (`Scalar+[V]`) every `n_stride` stiffness-scaled gradient steps."""
    def flow_step(current_vertices: Point, _: None) -> tuple[Point, Scalar]:
        edges = topology.boundary[2, 1] * (topology.d[0] * current_vertices)    # [3] Point-[F]
        angles = corner_angles(topology, edges)                                 # [3] Scalar+[F]
        normals, spreads = normal_spread(topology, edges, angles)               # Patch-[F], (Point- <- Point-)[V]
        eigenvalues, directions = spreads.batch().eigh()                        # [V, 3] Scalar+, [V, 3] Point-
        # Convert the stiffness-scaled gradient covector into an intrinsic vertex step via the metric dual.
        step = (energy_gradient(topology, edges, angles, normals, directions[:, 0].field()) / vertex_stiffness(topology, edges, angles)).dual()  # Point-[V]
        return current_vertices - step_scale * step, eigenvalues[:, 0]

    stride_scan = jax.jit(lambda current_vertices: jax.lax.scan(flow_step, current_vertices, length=n_stride))
    vertices = vertices.materialize(JaxContext(ga))                             # Point-[V]
    for _ in range(n_states):
        next_vertices, step_values = stride_scan(vertices)                      # Point-[V], [n_stride] Scalar+
        yield vertices.materialize(mv.context), step_values[0].materialize(mv.context).field()
        vertices = next_vertices
