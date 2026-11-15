"""Whitney forms reconstruct continuous fields across triangles and tetrahedra from vertex, edge, and face values.

Each triangle's signed area patch determines both the constant corner simplex gradients and the linear edge 1-forms without a metric. Assembling those local maps across a triangulation commutes with the coboundary operators and pulls material conductivity back to vertex and edge conductances through the adjugate.

In 3D projective space, joining the corners off a sub-simplex and dividing by all four corners joined produces the tetrahedral Whitney forms of every degree in one quotient, whose volume helicity pairing links only opposite edges.
"""

from __future__ import annotations

from functools import reduce
from operator import and_

import numpy as np

from numga import NumpyContext, stack
from numga.algebras import VGA2D as ga, PGA3D as space
from numga.extensor import Extensor
from numga.sparse import SparseExtensor, spdiag
from numga.topology import Topology

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                                                  # Scalar+
Plane = ga.gatype.vector()                                                   # Vector+
Point = ga.gatype.antivector()                                               # Antivector-
Patch = ga.gatype.antibivector()                                             # Antibivector-
Conductivity = ga.gatype((Point, Plane))                                     # Point- <- Plane+

# The tetrahedron's projective space gives corners unit weight so their joins span planes, lines, and points.
space_mv = NumpyContext(space).multivector
SpaceScalar = space.gatype.scalar()                                          # Scalar+
SpacePoint = space.gatype.antivector()                                       # Antivector-


def simplex_gradients(topology: Topology, vertices: Point) -> tuple[Patch, Plane]:
    """Each triangle's boundary edges determine its signed area patch (`Patch-[F]`) and three corner simplex gradients (`[3] Plane+[F]`)."""
    sides = topology.boundary[2, 1] * (topology.d[0] * vertices)             # [3] Point-[F]
    patches = (sides[0] & sides[1]) / 2                                      # Patch-[F]
    # Dividing each opposite edge by twice the signed area patch gives the simplex gradient
    # that rises by one from that edge to its corner without using a metric.
    gradients = sides / (2 * patches)                                        # [3] Plane+[F]
    return patches, gradients


def vertex_whitney_forms(topology: Topology, vertices: Point, points: Point) -> Scalar:
    """Each corner's simplex gradient evaluates the affine vertex 0-forms (`[3, ...] Scalar+[F]`) across `points`."""
    corners = topology.incidence[2, 0] * vertices                            # [3] Point-[F]
    patches, gradients = simplex_gradients(topology, vertices)               # Patch-[F], [3] Plane+[F]
    return 1 + (gradients[:, None] & (points[None, :] - corners[:, None]))   # [3, ...] Scalar+[F]


def edge_whitney_forms(topology: Topology, vertices: Point, points: Point) -> Plane:
    """Radial displacements from each opposite corner over twice the signed patch define the three local edge Whitney 1-forms (`[3, ...] Plane+[F]`) at `points`."""
    corners = topology.incidence[2, 0] * vertices                            # [3] Point-[F]
    sides = topology.boundary[2, 1] * (topology.d[0] * vertices)             # [3] Point-[F]
    patches = (sides[0] & sides[1]) / 2                                      # Patch-[F]
    # Along the two edges meeting at a corner, the radial displacement from that corner is parallel
    # to the edge, while along the opposite edge its join with the side equals twice the patch.
    return (points[None, :] - corners[:, None]) / (2 * patches)              # [3, ...] Plane+[F]


def edge_whitney_map(topology: Topology, vertices: Point, points: Point) -> SparseExtensor:
    """Local edge 1-forms weighted by oriented face boundaries assemble the evaluation map (`[...] (Plane+ <- Scalar+)[F <- E]`) at `points`."""
    forms = edge_whitney_forms(topology, vertices, points)                   # [3, ...] Plane+[F]
    boundaries = topology.boundary[2, 1][:, None]                            # [3, 1] Scalar+[F <- E]
    return (spdiag(forms) * boundaries).sum(axis=0) * Scalar                 # [...] (Plane+ <- Scalar+)[F <- E]


def tetrahedral_forms(topology: Topology, vertices: SpacePoint, degree: int) -> Extensor:
    """Dividing the join of the corners off each `degree`-simplex by all four corners joined gives the tetrahedral Whitney forms (`[...] Extensor+[T]`)."""
    # Reordering the off-simplex corners flips numerator and denominator alike and cancels.
    complements = reduce(and_, topology.off[3, degree] * vertices)
    spans = reduce(and_, topology.on[3, degree] * vertices)
    return complements / (complements & spans)


def tetrahedral_helicity(topology: Topology, vertices: SpacePoint) -> SpaceScalar:
    """Each edge's Whitney 1-form paired at the centroid with each edge's constant curl integrates to the symmetric volume helicity (`[6, 6] Scalar+[T]`)."""
    corners = topology.incidence[3, 0] * vertices                            # [4] Point-[T]
    forms = tetrahedral_forms(topology, vertices, 1)                         # [6] Bivector+[T]
    # Each edge 1-form is linear across the tetrahedron while its curl equals twice the form and stays constant,
    # so evaluating the 1-form at the centroid integrates their wedge product over the tetrahedron exactly.
    curls = 2 * forms                                                        # [6] Bivector+[T]
    centroid_forms = forms & corners.mean(axis=0)                            # [6] Plane+[T]
    densities = ((centroid_forms[:, None] ^ curls[None, :]) + (centroid_forms[None, :] ^ curls[:, None])) / 2  # [6, 6] Trivector+[T]
    volume = reduce(and_, corners[1:] - corners[0]) / 6                      # Volume-[T]
    return densities & volume                                                # [6, 6] Scalar+[T]


def tetrahedral_edge_pages(topology: Topology, vertices: SpacePoint, n_pages: int) -> SpacePoint:
    """Evenly spaced points along each edge span `n_pages` level-surface triangles (`[6, n_pages, 3] Point-[T]`) with the opposite edge."""
    corners = topology.incidence[3, 0] * vertices                            # [4] Point-[T]
    tails, heads = topology.corners[3, 1]
    opposite_tails, opposite_heads = topology.complements[3, 1]
    fractions = np.arange(1, n_pages + 1) / (n_pages + 1)                    # [n_pages]
    points = corners[tails, None] * (1 - fractions) + corners[heads, None] * fractions  # [6, n_pages] Point-[T]
    # At any point, the edge 1-form evaluates to the plane through that point and the opposite edge,
    # so every triangle spanned by the opposite edge and an edge point is a level surface of the form.
    opposite_tail = corners[opposite_tails, None].broadcast_to(points.shape)  # [6, n_pages] Point-[T]
    opposite_head = corners[opposite_heads, None].broadcast_to(points.shape)  # [6, n_pages] Point-[T]
    return stack((opposite_tail, points, opposite_head), axis=-1)            # [6, n_pages, 3] Point-[T]


def tetrahedral_face_lines(topology: Topology, vertices: SpacePoint, face_weights: np.ndarray) -> SpacePoint:
    """Straight segments from each tetrahedral corner to equal-area subtriangle centres on the opposite face trace the face 2-form field lines (`[4, ..., 2] Point-[T]`)."""
    corners = topology.incidence[3, 0] * vertices                            # [4] Point-[T]
    facets = corners[topology.complements[3, 0].T]                           # [4, 3] Point-[T]
    crossings = (facets[:, None] * face_weights).sum(axis=-1)                # [4, ...] Point-[T]
    # Evaluating a face 2-form at any point yields the line through the opposite corner and that point,
    # so straight segments from the opposite corner carry equal shares of the uniform face flux.
    return stack((corners[:, None].broadcast_to(crossings.shape), crossings), axis=-1)  # [4, ..., 2] Point-[T]


def simplex_gradient_map(topology: Topology, vertices: Point) -> SparseExtensor:
    """Corner simplex gradients summed by face incidence map vertex potentials to piecewise-constant face gradients (`(Plane+ <- Scalar+)[F <- V]`)."""
    patches, gradients = simplex_gradients(topology, vertices)               # Patch-[F], [3] Plane+[F]
    return (spdiag(gradients) * topology.incidence[2, 0]).sum(axis=0) * Scalar  # (Plane+ <- Scalar+)[F <- V]


def cochain_maps(topology: Topology, vertices: Point) -> tuple[SparseExtensor, SparseExtensor, SparseExtensor]:
    """Corner simplex gradients and signed patches assemble the centroid edge 1-form map (`(Plane+ <- Scalar+)[F <- E]`), exterior curl (`(Bivector+ <- Scalar+)[F <- E]`), and face 2-form map (`(Bivector+ <- Scalar+)[F]`)."""
    patches, gradients = simplex_gradients(topology, vertices)               # Patch-[F], [3] Plane+[F]
    tails, heads = topology.corners[2, 1]
    # At each face centroid all three vertex 0-forms equal 1/3, reducing each edge 1-form
    # to one third of the difference between its endpoint simplex gradients.
    centroid_forms = (gradients[heads] - gradients[tails]) / 3               # [3] Plane+[F]
    whitney1 = (spdiag(centroid_forms) * topology.oriented[2, 1]).sum(axis=0) * Scalar  # (Plane+ <- Scalar+)[F <- E]
    curls = 2 * (gradients[tails] ^ gradients[heads])                        # [3] Bivector+[F]
    curl = (spdiag(curls) * topology.oriented[2, 1]).sum(axis=0) * Scalar    # (Bivector+ <- Scalar+)[F <- E]
    whitney2 = mv.xy / (mv.xy & patches) * Scalar                            # (Bivector+ <- Scalar+)[F]
    return whitney1, curl, whitney2


def steady_conduction(
    topology: Topology,
    vertices: Point,
    conductivity: Conductivity,
    electrodes: np.ndarray,
    prescribed: Scalar,
) -> tuple[Scalar, Point]:
    """Pulling area-weighted conductivity back through the simplex gradient determines the steady vertex potential (`[...] Scalar+[V]`) and face current (`[...] Point-[F]`) for `prescribed` electrode potentials."""
    V, E, F = topology.n_elements
    patches, corner_gradients = simplex_gradients(topology, vertices)        # Patch-[F], [3] Plane+[F]
    areas = patches.antinorm()                                               # Scalar+[F]
    gradient = (spdiag(corner_gradients) * topology.incidence[2, 0]).sum(axis=0) * Scalar  # (Plane+ <- Scalar+)[F <- V]
    # Pulling the area-weighted conductivity back through the simplex gradient gives the vertex conductance.
    medium = areas * conductivity                                            # [...] (Point- <- Plane+)[F]
    conductance = gradient.adjugate()(medium(gradient))                      # [...] (Antiscalar- <- Scalar+)[V <- V]
    free_vertices = np.setdiff1d(np.arange(V), electrodes)
    free = topology.reading(free_vertices, 0).adjugate() * Scalar            # (Scalar+ <- Scalar+)[V <- V_free]
    rhs = -free.adjugate()(conductance(prescribed))                          # [...] Antiscalar-[V_free]
    potential = prescribed + free(free.adjugate()(conductance(free)).solve(rhs))  # [...] Scalar+[V]
    current = -conductivity(gradient(potential))                             # [...] Point-[F]
    return potential, current


def edge_conductance(topology: Topology, vertices: Point, conductivity: Conductivity) -> SparseExtensor:
    """Three-midpoint quadrature integrates quadratic products of linear edge 1-forms exactly into the edge conductance (`[...] (Antiscalar- <- Scalar+)[E <- E]`)."""
    patches, corner_gradients = simplex_gradients(topology, vertices)        # Patch-[F], [3] Plane+[F]
    areas = patches.antinorm()                                               # Scalar+[F]
    midpoints = topology.incidence[2, 1] * (topology.incidence[1, 0].mean(axis=0) * vertices)  # [3] Point-[F]
    midpoint_map = edge_whitney_map(topology, vertices, midpoints)           # [3] (Plane+ <- Scalar+)[F <- E]
    weights = ((areas / 3) * conductivity)[..., None]                        # [..., 3] (Point- <- Plane+)[F]
    return midpoint_map.adjugate()(weights(midpoint_map)).sum(axis=-1)       # [...] (Antiscalar- <- Scalar+)[E <- E]
