"""Garland–Heckbert quadric error metrics simplify a triangulated surface by collapsing edges to vertices that minimize squared distances to incident face planes.

In 3D projective geometric algebra, a plane is modelled as a vector and a point is modelled as an antivector (`Point`), so their regressive pairing `plane & point` gives the point's signed distance from a unit plane. A plane's dyad `plane * (plane & Point)` maps points to polar planes, and pairing that polar plane back with the point evaluates the squared distance. The error quadric of a vertex is the sum of its incident triangles' plane dyads, and collapsing an edge adds the quadrics of its two endpoints. Solving for a spatial direction offset `Direction` from the edge's midpoint finds the minimum at unit homogeneous weight and selects the minimizer nearest the midpoint when a flat patch or crease leaves directions free, following the quadric error metric of Garland and Heckbert, https://www.cs.cmu.edu/~garland/quadrics/.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import NumpyContext
from numga.algebras import PGA3D, VGA3D
from numga.topology import Topology
from examples.surfaces.qem import plumbing

ga = PGA3D
mv = NumpyContext(ga).multivector

Point = ga.gatype.antivector()                      # Antivector-
Plane = ga.gatype.vector()                          # Vector+
# A quadric maps each point to its polar plane.
Quadric = ga.gatype((Plane, Point))
Spatial = Plane.derive.within(VGA3D)                # Vector+
Direction = Spatial.derive.dual                     # Antivector-


# --- math -----------------------------------------------------------------------------
def collapse_position(quadric: Quadric, midpoint: Point) -> Point:
    """The unit-weight minimum of `quadric` nearest `midpoint` resolves unconstrained flat or crease directions (`[...] Point-`)."""
    # Displacing by `Direction` keeps unit homogeneous weight; zeroing the spatial polar plane leaves the minimum error in `w`.
    gradient = quadric(Direction).cast(Spatial)                                 # [...] Plane+ <- Direction-
    residual = quadric(midpoint).cast(Spatial)                                  # [...] Plane+
    correction = gradient.lstsq(-residual)                                      # [...] Direction-
    return midpoint + correction                                                # [...] Point-


def simplify(topology: Topology, vertices: Point, targets: np.ndarray) -> Iterator[tuple[Topology, Point]]:
    """Yields a closed triangulated surface and its quadric-optimal contractions at descending triangle-count `targets`.

    Merged vertices accumulate the quadrics of their initial predecessors so errors remain measured against the initial face planes, while each pass contracts disjoint neighbourhoods that preserve orientation and manifold connectivity.
    """
    corners = topology.incidence[2, 0] * vertices                               # [3] Point-[F]
    planes = (corners[0] & corners[1] & corners[2]).normalized()                # Plane-[F]
    face_quadrics = planes * (planes & Point)                                   # (Plane+ <- Point-)[F]
    # Sum incident face quadrics onto each vertex via the adjugate incidence map.
    quadrics = topology.incidence[2, 0].sum(axis=0).adjugate() * face_quadrics  # (Plane+ <- Point-)[V]
    yield topology, vertices

    for target in targets:
        V, E, F = topology.n_elements
        while F > target:
            ends = topology.incidence[1, 0]                                     # [2] Scalar+[E <- V]
            edge_quadrics = ends.sum(axis=0) * quadrics                         # (Plane+ <- Point-)[E]
            midpoints = ends.mean(axis=0) * vertices                            # Point-[E]
            # Solve each edge's 3x3 least-squares problem independently across the batch of edges.
            optima = collapse_position(edge_quadrics.batch(), midpoints.batch()).field()    # Point-[E]
            errors = edge_quadrics(optima) & optima                             # Scalar+[E]

            # Test every non-degenerate triangle moved by an edge contraction to ensure its plane keeps its orientation and nonzero area.
            corners = topology.incidence[2, 0] * vertices                       # [3] Point-[F]
            tail, head = ends * vertices                                        # Point-[E], Point-[E]
            # Each corner's opposite edge oriented along the triangle's boundary; joining it with the trial vertex forms the moved triangle's plane.
            facing_sides = topology.boundary[2, 1] * (tail & head)              # [3] Line-[F]
            at_edges, at_corners = plumbing.trials(topology)
            old_planes = at_corners.sum(axis=0) * (corners[0] & corners[1] & corners[2])    # Plane-[n_trials]
            new_planes = (at_corners * facing_sides).sum(axis=0) & (at_edges * optima)      # Plane-[n_trials]
            valid_trials = (old_planes.scalar_product(new_planes) > 0) & (new_planes.squared() > old_planes.squared() * 1e-12)
            legal = at_edges.adjugate()(~valid_trials) <= 0                     # [E]
            chosen_edges = plumbing.select_edges(topology, errors, legal, max(1, (F - target) // 2))

            # Selected vertex stars share no moved triangles, so their contractions apply simultaneously without invalidating the trial checks.
            topology, kept, merged = plumbing.contract(topology, chosen_edges)
            vertices = kept * vertices + merged * optima                        # Point-[V_coarser]
            quadrics = kept * quadrics + merged * edge_quadrics                 # (Plane+ <- Point-)[V_coarser]
            V, E, F = topology.n_elements
        yield topology, vertices
