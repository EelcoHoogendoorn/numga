"""A planar triangulation lifted by vertex heights into a polyhedral terrain, its bar stresses, and its reciprocal diagram.

Vertex heights lift a planar triangulation into a polyhedral terrain that serves as a piecewise-linear Airy stress function. Vertices are intrinsic `Point-` antivectors and each terrain face is an extrinsic `Plane+` vector whose pairing `plane & (point + up)` with a point lifted to unit height reads the height there. Because the three simplex gradients of a triangle's lifted corners pair to one with their own corner and to zero with the opposite two, the face plane is linear in the corner heights. Across each interior edge the terrain creases along a bar of the planar truss, and the slope jump per unit bar length gives the bar stress, positive in tension where the crease bends convex. The face slopes form the vertices of the reciprocal diagram, whose edges cross the bars perpendicularly with lengths equal to the bar forces so every interior vertex stays in equilibrium while the boundary carries the external loads. Lifted onto the paraboloid, the reciprocal diagram becomes the Voronoi diagram and the bar stresses equal the cotangent weights.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import NumpyContext
from numga.algebras import VGA3D as ga
from numga.topology import Topology

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                         # Scalar+
# Each terrain face is an extrinsic plane vector, and each vertex is an intrinsic point antivector.
Plane = ga.gatype.vector()                          # Vector+
Point = ga.gatype.antivector()                      # Antivector-
# Unit upward point offset that lifts planar points to unit height.
up = mv.z.dual()                                    # [] Point-


# --- math -----------------------------------------------------------------------------
def face_planes(topology: Topology, points: Point, heights: Scalar) -> Plane:
    """The plane of each triangle face (`[...] Plane+[F]`) passes through its corners at `heights` (`[...] Scalar+[V]`) and pairs with `point + up` to the terrain height there."""
    lifted = points + up                                                       # Point-[V]
    corners = topology.incidence[2, 0] * lifted                                # [3] Point-[F]
    facing = topology.off[2, 0] * lifted                                       # [2] Point-[F]
    # Joining the two opposite corners gives the plane through them and the origin; dividing by its pairing with the third corner normalizes each simplex gradient to one there.
    sides = facing[0] & facing[1]                                              # [3] Plane+[F]
    gradients = sides / (sides & corners)                                      # [3] Plane+[F]
    corner_heights = topology.incidence[2, 0] * heights[..., None]             # [..., 3] Scalar+[F]
    return (gradients * corner_heights).sum(axis=-1)                           # [...] Plane+[F]


def bar_stresses(topology: Topology, points: Point, heights: Scalar) -> Scalar:
    """The axial stress on each interior bar (`[...] Scalar+[E_interior]`) equals the slope jump between its adjacent faces per unit bar length, positive in tension where the crease bends convex."""
    interior = topology.reading(topology.interior[1], 1)                       # Scalar+[E_interior <- E]
    # Pulling face planes back onto interior edges subtracts the right-hand face plane from the left-hand face plane along each oriented bar.
    jumps = interior * (topology.d[1].adjugate() * face_planes(topology, points, heights))  # [...] Plane+[E_interior]
    bars = interior * (topology.d[0] * points)                                 # Point-[E_interior]
    tails = (interior * (topology.incidence[1, 0] * points))[0]                # Point-[E_interior]
    # Project out the component along the bar to isolate the perpendicular offset toward the opposite right-hand corner.
    far_corners = topology.reading(topology.facet_opposites[:, 1], 0) * points  # Point-[E_interior]
    offsets = far_corners - tails                                              # Point-[E_interior]
    transverse = offsets - bars * ((bars | offsets) / (bars | bars))           # Point-[E_interior]
    return -(jumps & transverse) / (transverse.antinorm() * bars.antinorm())   # [...] Scalar+[E_interior]


def edge_folds(topology: Topology, points: Point, heights: Scalar) -> Scalar:
    """The vertical fold across each interior edge (`[...] Scalar+[E_interior]`) measures how far the opposite right-hand corner rises above the left-hand face plane."""
    interior = topology.reading(topology.interior[1], 1)                       # Scalar+[E_interior <- E]
    # Because the right-hand face plane passes through that opposite corner at its lifted height, pairing the slope jump at unit height reads the vertical clearance directly.
    jumps = interior * (topology.d[1].adjugate() * face_planes(topology, points, heights))  # [...] Plane+[E_interior]
    lifted_corners = topology.reading(topology.facet_opposites[:, 1], 0) * (points + up)    # Point-[E_interior]
    return -(jumps & lifted_corners)                                           # [...] Scalar+[E_interior]


def breathing_states(
    topology: Topology,
    points: Point,
    base_heights: Scalar,
    phases: np.ndarray,
    amplitude: float,
    n_frames: int,
) -> Iterator[tuple[Scalar, Scalar, Plane]]:
    """Yields `(heights, stresses, dual_vertices)` (`(Scalar+[V], Scalar+[E_interior], Plane+[F])`) across `n_frames` frames of one harmonic oscillation around `base_heights`."""
    for step in range(n_frames):
        angle = 2 * np.pi * step / n_frames
        heights = base_heights + mv.scalar((amplitude * np.sin(angle + phases))[:, None]).field()  # Scalar+[V]
        stresses = bar_stresses(topology, points, heights)                                         # Scalar+[E_interior]
        dual_vertices = (face_planes(topology, points, heights) ^ mv.z) * mv.z                     # Plane+[F]
        yield heights, stresses, dual_vertices
