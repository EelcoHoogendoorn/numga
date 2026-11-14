"""Keenan Crane's spin transformations, in geometric algebra.

https://www.cs.cmu.edu/~kmcrane/Projects/SpinTransformations/paper.pdf

A spin transformation changes a surface's mean curvature by a prescribed amount while keeping every
angle. A scaled rotor at each vertex, an even multivector, turns and scales the edges around it. The
rotors are the field of least energy, the area-weighted squared residual of the Dirac operator less the
curvature change, and the new vertices are those whose edges best match the turned ones. The vertices
and edges are intrinsic points, antivectors, which in three dimensions are bivectors. The Dirac
operator's couplings are bivectors, and the curvature change is an intrinsic scalar in the unit of
volume so that the two share their orientation, while the operators are sparse linear maps between
fields on the vertices, edges and faces, coupling their elements through multivectors.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import NumpyContext
from numga.algebras import VGA3D as ga
from numga.sparse import SparseExtensor, spdiag
from numga.topology import Topology

context = NumpyContext(ga)
mv = context.multivector

Scalar = ga.gatype.scalar()                                                    # Scalar+
Plane = ga.gatype.vector()                                                     # Vector+
Point = ga.gatype.antivector()                                                 # Antivector-
Patch = ga.gatype.antibivector()                                               # Antibivector-
Even = ga.gatype.even()                                                        # Even+
# A volume, the join of three points, is an intrinsic scalar, and its unit gives an ordinary number that orientation.
unit_volume = mv.xyz.dual()                                                    # [] Antipseudoscalar-


# --- math -----------------------------------------------------------------------------
def cotangent_weights(topology: Topology, sides: Point) -> Scalar:
    """Each edge's cotangent weight (`Scalar+[E]`) is half the sum of the cotangents of the two angles
    facing it, read from the edge vectors `sides` (`[3] Point-[F]`) facing each corner."""
    tails, heads = topology.corners[2, 1]
    after, before = sides[tails], sides[heads]                                 # [3] Point-[F], [3] Point-[F]
    cotangents = (after | before) / (after & before).norm()                    # [3] Scalar+[F]
    return (topology.incidence[2, 1].adjugate() * (cotangents / 2)).sum(axis=0)  # Scalar+[E]


def corner_cosines(topology: Topology, vertices: Point) -> Scalar:
    """Each triangle corner's angle cosine (`[3] Scalar+[F]`) comes from the inner product of the two
    edges meeting there over the product of their lengths."""
    sides = topology.boundary[2, 1] * (topology.d[0] * vertices)               # [3] Point-[F]
    tails, heads = topology.corners[2, 1]
    after, before = sides[tails], sides[heads]                                 # [3] Point-[F], [3] Point-[F]
    return (after | before) / (after.norm() * before.norm())                   # [3] Scalar+[F]


def deformed_surface(topology: Topology, vertices: Point, curvature_change: Scalar) -> Point:
    """The spin transformation for a prescribed `curvature_change` (`Scalar+[F]`) turns the edges by the
    rotor field of least Dirac residual energy per unit vertex area and solves for the vertices (`Point-[V]`)
    whose edges best match the turned ones."""
    edge_means = topology.incidence[1, 0].mean(axis=0)                         # Scalar+[E <- V]
    face_means = topology.incidence[2, 0].mean(axis=0)                         # Scalar+[F <- V]
    edges = topology.d[0] * vertices                                           # Point-[E]
    sides = topology.boundary[2, 1] * edges                                    # [3] Point-[F]
    areas = (sides[0] & sides[1]).norm() / 2                                   # Scalar+[F]
    face_areas, vertex_areas = spdiag(areas), spdiag(face_means.adjugate() * areas)  # Scalar+[F <- F], Scalar+[V <- V]
    edge_weights = spdiag(cotangent_weights(topology, sides))                  # Scalar+[E <- E]
    laplacian = topology.d[0].adjugate() * edge_weights * topology.d[0]        # Scalar+[V <- V]

    residual = (spdiag(sides / (2 * areas) - curvature_change * unit_volume / 3) * topology.incidence[2, 0]).sum(axis=0)  # Even-[F <- V]
    _, (field,) = ((~residual * face_areas * residual) * Even).eigh(vertex_areas * Even, 1)  # Even+[V]
    rotors = field / ((vertex_areas * field).sites.sum() / areas.sites.sum())  # Even+[V]
    turned_edges = ((edge_means * (rotors << Point))(edges) + 2 * ((edge_means * rotors) << edges)) / 3  # Point-[E]
    return (laplacian * Point).lstsq(topology.d[0].adjugate() * edge_weights * turned_edges)  # Point-[V]


def dirac_spheres(topology: Topology, vertices: Point, eigenvalues: np.ndarray, n_states: int) -> tuple[Scalar, Point]:
    """Each constant curvature shift in `eigenvalues` (`[n_eigenvalues]`) gives the `n_states` rotor
    eigenfields of least Dirac residual energy (`[n_eigenvalues, n_states] Scalar+`) and spins the
    surface into the corresponding Dirac spheres (`[n_eigenvalues, n_states] Point-[V]`)."""
    edge_means = topology.incidence[1, 0].mean(axis=0)                         # Scalar+[E <- V]
    face_means = topology.incidence[2, 0].mean(axis=0)                         # Scalar+[F <- V]
    edges = topology.d[0] * vertices                                           # Point-[E]
    sides = topology.boundary[2, 1] * edges                                    # [3] Point-[F]
    areas = (sides[0] & sides[1]).norm() / 2                                   # Scalar+[F]
    face_areas, vertex_areas = spdiag(areas), spdiag(face_means.adjugate() * areas)  # Scalar+[F <- F], Scalar+[V <- V]
    edge_weights = spdiag(cotangent_weights(topology, sides))                  # Scalar+[E <- E]
    laplacian = topology.d[0].adjugate() * edge_weights * topology.d[0]        # Scalar+[V <- V]

    shifts = eigenvalues[:, None] * unit_volume / 3                            # [..., 1] Antipseudoscalar-
    residual = (spdiag(sides / (2 * areas) - shifts) * topology.incidence[2, 0]).sum(axis=-1)  # [...] Even-[F <- V]
    energies, eigenfields = ((~residual * face_areas * residual) * Even).eigh(vertex_areas * Even, n_states)  # [..., n_states] Even+[V]
    turned_edges = ((edge_means * (eigenfields << Point))(edges) + 2 * ((edge_means * eigenfields) << edges)) / 3  # [..., n_states] Point-[E]
    return energies, (laplacian * Point).lstsq(topology.d[0].adjugate() * edge_weights * turned_edges)


def geometric_derivative(topology: Topology, vertices: Point) -> SparseExtensor:
    """The geometric derivative (`Plane+[F <- V]`) of piecewise-linear vertex fields couples each face
    to each of its corners through that corner's simplex gradient, the opposite edge contracted with
    the inverse of twice the face's oriented patch."""
    sides = topology.boundary[2, 1] * (topology.d[0] * vertices)               # [3] Point-[F]
    patches = (sides[0] & sides[1]) / 2                                        # Patch-[F]
    gradients = (sides | patches.inverse()) / 2                                # [3] Plane+[F]
    return (spdiag(gradients) * topology.incidence[2, 0]).sum(axis=0)          # Plane+[F <- V]


def mean_curvature(topology: Topology, vertices: Point) -> Scalar:
    """Each face's signed mean curvature (`Scalar+[F]`) averages the vertex mean curvatures, where the
    cotangent Laplacian of the positions gives the area-integrated mean-curvature normal met with the
    unit vertex normal over twice the vertex area."""
    face_means = topology.incidence[2, 0].mean(axis=0)                         # Scalar+[F <- V]
    sides = topology.boundary[2, 1] * (topology.d[0] * vertices)               # [3] Point-[F]
    patches = (sides[0] & sides[1]) / 2                                        # Patch-[F]
    areas = patches.norm()                                                     # Scalar+[F]
    laplacian = topology.d[0].adjugate() * spdiag(cotangent_weights(topology, sides)) * topology.d[0]  # Scalar+[V <- V]
    # Summed face patches around each vertex give an area-weighted normal once normalized.
    normals = (face_means.adjugate() * patches).normalized()                   # Patch-[V]
    # Meeting the integrated mean-curvature normal with the vertex normal yields an intrinsic volume scalar.
    integrated_curvature = ((laplacian * vertices) & normals) * unit_volume    # Scalar+[V]
    return face_means * (integrated_curvature / (2 * (face_means.adjugate() * areas)))  # Scalar+[F]


def recentred_vertices(vertices: Point) -> Point:
    """Centring `vertices` (`Point-[V]`) and scaling to unit mean radius fixes the translation and
    overall scale that a spin transformation leaves free."""
    centred = vertices - vertices.sites.mean()                                 # Point-[V]
    return centred / centred.norm().sites.mean()                               # Point-[V]


def surface_roundness(topology: Topology, vertices: Point) -> Scalar:
    """The relative spread (`[] Scalar+`) of vertex distances from the surface's area-weighted centroid,
    which vanishes on any sphere regardless of vertex crowding."""
    centroids = topology.incidence[2, 0].mean(axis=0) * vertices               # Point-[F]
    sides = topology.boundary[2, 1] * (topology.d[0] * vertices)               # [3] Point-[F]
    areas = (sides[0] & sides[1]).norm() / 2                                   # Scalar+[F]
    middle = (centroids * areas).sites.sum() / areas.sites.sum()               # [] Point-
    distances = (vertices - middle).norm()                                     # Scalar+[V]
    mean_distance = distances.sites.mean()                                     # [] Scalar+
    return (distances - mean_distance).squared().sites.mean().square_root() / mean_distance  # [] Scalar+


def curvature_flow_states(topology: Topology, vertices: Point, n_steps: int, rate: float) -> Iterator[Point]:
    """Yields the surface vertices (`Point-[V]`) at the start and after each of `n_steps` steps of
    conformal curvature flow, where each step removes a share `rate` of every face's departure from
    the area-weighted mean curvature."""
    vertices = recentred_vertices(vertices)                                    # Point-[V]
    yield vertices
    for _ in range(n_steps):
        curvature = mean_curvature(topology, vertices)                         # Scalar+[F]
        sides = topology.boundary[2, 1] * (topology.d[0] * vertices)           # [3] Point-[F]
        areas = (sides[0] & sides[1]).norm() / 2                               # Scalar+[F]
        mean = (curvature * areas).sites.sum() / areas.sites.sum()             # [] Scalar+
        vertices = recentred_vertices(deformed_surface(topology, vertices, -rate * (curvature - mean)))  # Point-[V]
        yield vertices
