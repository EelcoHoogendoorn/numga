"""Every triangulation of the sphere is the incidence pattern of a Koebe polyhedron whose edges all touch the unit sphere.

In the algebra whose metric is the unit sphere, spatial planes square to one and the ideal plane squares to minus one. Each circle where a plane cuts the sphere is a unit vector, and each point is an antivector that lies on the sphere when it is null and on a circle when `circle & point == 0`. Dualizing a circle gives its pole, the apex of the cone tangent to the sphere along that circle. Two circles touch externally when `circle | other == -1`, and the sum of their poles `circle.dual() + other.dual()` is the null tangency point on the edge between their apexes. Dualizing the wedge of three vertex circles around a face `(a ^ b ^ c).dual()` gives the orthogonal face circle through those three tangency points, whose apex is a vertex of the polar dual polyhedron touching the sphere at the same points. Rotors preserve every inner product and carry any Koebe polyhedron through a six-parameter family of shapes while keeping every edge tangent to the sphere; centring places the tangency points' balance centre at the centre of the sphere.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import Algebra, NumpyContext, stack
from numga.sparse import Sparse, spdiag
from numga.topology import Topology

ga = Algebra("x+y+z+w-")
mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                                                 # Scalar+
# Extrinsic planes cutting the unit sphere along oriented circles.
Plane = ga.gatype.vector()                                                  # Vector+
# Intrinsic face circles dual to three wedged vertex planes.
Patch = Plane.derive.flipped_orientation                                    # Vector-
# Points of space and null points of the sphere, dual to polar planes.
Point = ga.gatype.antivector()                                              # Antivector-
Rotor = ga.gatype.rotor()                                                   # Rotor+
# Centre of the unit sphere, dual to the weight plane w.
origin = mv.w.dual()                                                        # [] Point-


def cap_circles(topology: Topology, directions: Point) -> Plane:
    """Spherical caps around `directions` (`Point-[V]`) of radius half the mean edge angle initialize the circle planes (`Plane+[V]`)."""
    tails, heads = topology.incidence[1, 0] * directions                    # Point-[E], Point-[E]
    angles = (heads * tails).log().norm()                                   # Scalar+[E]
    radius = angles.sites.mean() / 2                                        # [] Scalar+
    # Place each cone apex along its direction at reciprocal cosine distance so its dual circle plane squares to one.
    apexes = (directions + origin * radius.cos()) / radius.sin()            # Point-[V]
    return apexes.dual()                                                    # Plane+[V]


def koebe_circles(topology: Topology, start: Plane, n_steps: int) -> Plane:
    """Gauss-Newton iterations from `start` (`Plane+[V]`) solve for unit circles touching externally along every edge (`Plane+[V]`)."""
    V, E, F = topology.n_elements
    vertex_rows = Sparse.selection(np.arange(V), V + E).adjugate()          # Scalar+[V + E <- V]
    edge_rows = Sparse.selection(V + np.arange(E), V + E).adjugate()        # Scalar+[V + E <- E]
    circles = start                                                         # Plane+[V]
    for _ in range(n_steps):
        tails, heads = topology.incidence[1, 0] * circles                   # Plane+[E], Plane+[E]
        residual = vertex_rows * (circles.squared() - 1) + edge_rows * ((tails | heads) + 1)  # Scalar+[V + E]
        # Linearize unit-square residuals on vertices and tangency inner products across incident edge endpoints.
        vertex_part = spdiag(2 * circles | Plane)                           # (Scalar+ <- Plane+)[V <- V]
        edge_part = (spdiag(stack([heads, tails]) | Plane) * topology.incidence[1, 0]).sum(axis=0)  # (Scalar+ <- Plane+)[E <- V]
        jacobian = vertex_rows * vertex_part + edge_rows * edge_part        # (Scalar+ <- Plane+)[V + E <- V]
        # Minimum-norm steps in the circle metric commute with sphere-preserving rotors and leave their six null directions untouched.
        circles = circles - jacobian.adjoint()(jacobian(jacobian.adjoint()).solve(residual))
    return circles


def tangency_points(topology: Topology, circles: Plane) -> Point:
    """Summing the poles at each edge's endpoints locates the null tangency points on the sphere (`[...] Point-[E]`)."""
    return topology.incidence[1, 0].sum(axis=0) * circles.dual()            # [...] Point-[E]


def cone_apexes(circles: Plane | Patch) -> Point:
    """Dualizing `circles` and normalizing by `w` places the tangent-cone apexes at unit weight (`[...] Point-`)."""
    poles = circles.dual()                                                  # [...] Antivector+-
    return poles / (mv.w & poles)                                           # [...] Point-


def face_circles(topology: Topology, circles: Plane) -> Patch:
    """Dualizing the wedge of each triangle's three vertex circles gives the orthogonal face circles (`[...] Patch-[F]`)."""
    corners = topology.incidence[2, 0] * circles[..., None]                 # [..., 3] Plane+[F]
    return (corners[..., 0] ^ corners[..., 1] ^ corners[..., 2]).dual()     # [...] Patch-[F]


def balance_centre(points: Point, n_iterations: int) -> Point:
    """Fixed-point iteration inside the sphere finds the unit-timelike point (`[] Point-`) minimizing summed log inner products with `points`."""
    # At the stationary point, weighting each null point by minus the reciprocal of its inner product sums parallel to the centre.
    centre = origin                                                         # [] Point-
    for _ in range(n_iterations):
        weighted_sum = (points * (-1 / (centre | points))).sites.sum()      # [] Point-
        centre = weighted_sum.normalized()
    return centre


def boost_rotors(rapidity: float, tilt: float, n_frames: int) -> Rotor:
    """Boost rotors (`[n_frames] Rotor+`) sweep a plane through `w` tilted by `tilt` once around the `z`-axis up to `rapidity`."""
    turn_angles = np.linspace(0, 2 * np.pi, n_frames, endpoint=False)       # [n_frames]
    tilted_plane = (mv.zx * (tilt / 2)).exp() >> mv.xw                      # [] Bivector+
    boost_planes = (mv.xy * (-turn_angles / 2)).exp() >> tilted_plane       # [n_frames] Bivector+
    return (boost_planes * (rapidity * np.sin(turn_angles) ** 2 / 2)).exp()  # [n_frames] Rotor+


def boost_states(circles: Plane, rapidity: float, tilt: float, n_frames: int) -> Iterator[Plane]:
    """Yields `circles` (`Plane+[V]`) carried across `n_frames` by `boost_rotors(rapidity, tilt, n_frames)`."""
    yield from boost_rotors(rapidity, tilt, n_frames) >> circles            # [n_frames] Plane+[V]
