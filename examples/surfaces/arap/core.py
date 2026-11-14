"""As-rigid-as-possible surface deformation bends a triangulated surface to follow displaced handle vertices while keeping every local neighbourhood close to a rigid rotation of its rest shape.

The optimal rotor at each vertex comes from a symmetric bilinear form on rotors: `edges.dot(Rotor >> rest_edges)` has type `Scalar+ <- (Rotor+, Rotor+)` satisfying `form(R, R) == edges.dot(R >> rest_edges)`, and summing those forms onto vertices yields every vertex's rotor as the top eigenvector in one batched `eigh` call without singular value decompositions or reflection sign fixes. The updated vertex positions then follow from the endpoint rotation maps `rotors >> Point` averaged across each edge, independent of each eigenvector's sign, through the cotangent Laplacian `T10.adjugate() * H1 * T10` in the local-global iteration of Sorkine and Alexa (2007), https://igl.ethz.ch/projects/ARAP/arap_web.pdf.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np

from numga import NumpyContext, stack
from numga.algebras import VGA3D
from numga.sparse import spdiag
from numga.topology import Topology

ga = VGA3D
mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                         # Scalar+
Point = ga.gatype.antivector()                      # Antivector-
Rotor = ga.gatype.rotor()                           # Rotor+


# --- math -----------------------------------------------------------------------------
def cotangent_weights(topology: Topology, rest_edges: Point) -> Scalar:
    """Half the cotangents of the two corners opposite each edge form the edge cotangent weights (`Scalar+[E]`)."""
    triangle_edges = topology.boundary[2, 1] * rest_edges                           # [3] Point-[F]
    tails, heads = topology.corners[2, 1]
    next_edges, prev_edges = triangle_edges[tails], triangle_edges[heads]           # [3] Point-[F], [3] Point-[F]
    cotangents = -(next_edges.dual_inverse() & prev_edges) / next_edges.commutator(prev_edges).norm()  # [3] Scalar+[F]
    return (topology.incidence[2, 1].adjugate() * (cotangents / 2)).sum(axis=0)     # Scalar+[E]


def corner_cosines(topology: Topology, vertices: Point) -> Scalar:
    """Normalized inner products of the two directed triangle sides meeting at each corner give the corner angle cosines (`[3] Scalar+[F]`)."""
    sides = topology.boundary[2, 1] * (topology.d[0] * vertices)                    # [3] Point-[F]
    tails, heads = topology.corners[2, 1]
    next_edges, prev_edges = sides[tails], sides[heads]                             # [3] Point-[F], [3] Point-[F]
    return -(next_edges.dual_inverse() & prev_edges) / (next_edges.antinorm() * prev_edges.antinorm())  # [3] Scalar+[F]


def local_rotors(edge_means, edge_weights, rest_edges: Point, edges: Point) -> Rotor:
    """Summing each edge's alignment quadratic form onto its endpoints yields the optimal vertex rotors (`Rotor+[V]`) as top eigenvectors."""
    forms = edge_means.adjugate() * edge_weights * edges.dot(Rotor >> rest_edges)   # (Scalar+ <- (Rotor+, Rotor+))[V]
    _, modes = forms.batch().eigh()                                                 # [..., 4] Rotor+
    return modes[..., -1].field()                                                   # Rotor+[V]


def global_vertices(edge_diffs, edge_means, edge_weights, handle_penalty, handle_laplacian,
                    rest_edges: Point, rotors: Rotor, targets: Point) -> Point:
    """Averaging the endpoint rotation maps across each edge and solving the handle-pinned cotangent Laplacian updates the deformed vertices (`Point-[V]`)."""
    turned_edges = (edge_means * (rotors >> Point))(rest_edges)                     # Point-[E]
    return handle_laplacian.solve(edge_diffs.adjugate() * edge_weights * turned_edges + handle_penalty * targets)  # Point-[V]


def deformation_states(
    topology: Topology,
    rest_vertices: Point,
    handles: np.ndarray,
    poses: Iterable[Point],
    stiffness: float,
    n_iterations: int,
) -> Iterator[tuple[Point, Point, Scalar]]:
    """Yields `(rigid_vertices, laplacian_vertices, energies)` for each handle pose across `n_iterations` warm-started local-global steps."""
    edge_diffs = topology.d[0]                                                      # Scalar+[E <- V]
    edge_means = topology.incidence[1, 0].mean(axis=0)                              # Scalar+[E <- V]
    rest_edges = edge_diffs * rest_vertices                                         # Point-[E]
    edge_weights = spdiag(cotangent_weights(topology, rest_edges))                  # Scalar+[E <- E]
    handle_penalty = spdiag(mv.scalar((handles * stiffness)[..., None]).field())    # Scalar+[V <- V]
    handle_laplacian = (edge_diffs.adjugate() * edge_weights * edge_diffs + handle_penalty) * Point  # (Point- <- Point-)[V <- V]

    vertices = rest_vertices                                                        # Point-[V]
    for targets in poses:
        energies = []
        for _ in range(n_iterations):
            edges = edge_diffs * vertices                                           # Point-[E]
            rotors = local_rotors(edge_means, edge_weights, rest_edges, edges)      # Rotor+[V]
            turned_edges = (edge_means * (rotors >> Point))(rest_edges)             # Point-[E]
            # Because each rotor preserves edge length, the two endpoint mismatches sum to 2 * (edge^2 + rest^2 - 2 * edge . turned).
            mismatch = edge_weights * (edges.norm_squared() + rest_edges.norm_squared() - 2 * edges.dot(turned_edges))  # Scalar+[E]
            penalty = handle_penalty * (vertices - targets).norm_squared()          # Scalar+[V]
            energies.append(2 * mismatch.sites.sum() + penalty.sites.sum())
            vertices = handle_laplacian.solve(edge_diffs.adjugate() * edge_weights * turned_edges + handle_penalty * targets)  # Point-[V]
        laplacian_vertices = handle_laplacian.solve(edge_diffs.adjugate() * edge_weights * rest_edges + handle_penalty * targets)  # Point-[V]
        yield vertices, laplacian_vertices, stack(energies)
