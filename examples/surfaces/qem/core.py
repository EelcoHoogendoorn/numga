"""Garland–Heckbert quadric error metrics simplify a triangle mesh by collapsing edges to vertices that minimize squared distances to incident face planes.

In 3D projective geometric algebra, a plane is modelled as a vector and a point is modelled as an antivector (`Point`), so their regressive pairing `plane & point` gives the point's signed distance from a unit plane. A plane's dyad `plane * (plane & Point)` maps points to polar planes, and pairing that polar plane back with the point evaluates the squared distance. The error quadric of a vertex is the sum of its incident triangles' plane dyads, and collapsing an edge adds the quadrics of its two endpoints. Solving for a spatial direction offset `Direction` from the edge's midpoint finds the minimum at unit homogeneous weight and selects the minimizer nearest the midpoint when a flat patch or crease leaves directions free, following the quadric error metric of Garland and Heckbert, https://www.cs.cmu.edu/~garland/quadrics/.

In homogeneous matrix notation each unit plane reads as a four-vector $\\mathbf p = (n_x, n_y, n_z, d)^{\\mathsf T}$, its plane dyad reads as the outer product $Q = \\mathbf p\\mathbf p^{\\mathsf T}$, and the squared distance at $\\mathbf x = (x, y, z, 1)^{\\mathsf T}$ reads as the quadratic form $\\mathbf x^{\\mathsf T} Q\\,\\mathbf x$.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga.algebras import PGA3D
from examples import instantiate
from examples.surfaces.qem import plumbing

ga = PGA3D
mesh_module = instantiate("examples.util.mesh", ga)
Mesh, as_scalar, mv = mesh_module.Mesh, mesh_module.as_scalar, mesh_module.mv

Point = ga.gatype.antivector()                      # Antivector-
Plane = ga.gatype.vector()                          # Vector+
# A quadric: the polar plane of each point.
Quadric = ga.gatype((Plane, Point))
Spatial = ga.gatype.from_blades("x y z")            # Vector+
Direction = Spatial.dual().output_gatype            # Antivector-


# --- math -----------------------------------------------------------------------------
def collapse_position(quadric: Quadric, midpoint: Point) -> Point:
    """The collapse position is the unit-weight minimum of the quadric, the one nearest the midpoint where flat directions leave a choice."""
    # A direction changes position without changing homogeneous weight. Only the spatial
    # part of the polar plane must vanish; its constant part is the residual minimum error.
    gradient = quadric(Direction).cast(Spatial)                                 # [...] Plane+ <- Direction-
    residual = quadric(midpoint).cast(Spatial)                                  # [...] Plane+
    correction = gradient.lstsq(-residual)                                      # [...] Direction-
    return midpoint + correction                                                # [...] Point-


def simplify(mesh: Mesh, vertices: Point, targets: np.ndarray) -> Iterator[tuple[Mesh, Point]]:
    """Yield a closed triangle mesh and its QEM contractions at descending face-count targets.

    The initial face planes remain the reference surface: every merged vertex retains the
    sum of its initial vertices' quadrics. Each pass contracts separate neighbourhoods,
    choosing the least errors first and preserving the topology and the sense of each face.
    """
    corners = mesh.incidence[2, 0] * vertices                                   # [corners] Point-[faces]
    planes = (corners[0] & corners[1] & corners[2]).normalized()                # Plane-[faces]
    face_quadrics = planes * (planes & Point)                                   # (Plane+ <- Point-)[faces]
    # Each vertex's quadric sums those of the faces at it.
    quadrics = ~mesh.incidence[2, 0].sum(axis=0) * face_quadrics                # (Plane+ <- Point-)[vertices]
    yield mesh, vertices

    for target in targets:
        while len(mesh.faces) > target:
            ends = mesh.incidence[1, 0]                                         # [ends] Scalar+[edges <- vertices]
            edge_quadrics = ends.sum(axis=0) * quadrics                         # (Plane+ <- Point-)[edges]
            midpoints = ends.mean(axis=0) * vertices                            # Point-[edges]
            # Each edge's least squares is its own, solved on the batch of edges.
            optima = collapse_position(edge_quadrics.batch(), midpoints.batch()).field()   # Point-[edges]
            errors = edge_quadrics(optima) & optima                             # Scalar+[edges]

            # Test every surviving triangle in an edge's neighbourhood with that edge
            # contracted. Its plane must keep its sense and a nonzero area.
            corners = mesh.incidence[2, 0] * vertices                           # [corners] Point-[faces]
            tail, head = ends * vertices                                        # Point-[edges], Point-[edges]
            # The side facing each corner, running with the face: joined with a point, the plane through it.
            facing_sides = mesh.boundary[2, 1] * (tail & head)                  # [corners] Line-[faces]
            at_edges, at_corners = plumbing.trials(mesh)
            old_planes = at_corners.sum(axis=0) * (corners[0] & corners[1] & corners[2])    # Plane-[trials]
            new_planes = (at_corners * facing_sides).sum(axis=0) & (at_edges * optima)     # Plane-[trials]
            valid_trials = (old_planes.scalar_product(new_planes) > 0) & (new_planes.squared() > old_planes.squared() * 1e-12)
            failures = ~at_edges * as_scalar(~valid_trials).field()             # Scalar+[edges]
            chosen_edges = plumbing.select_edges(mesh, errors, failures <= 0, max(1, (len(mesh.faces) - target) // 2))
            if not len(chosen_edges):
                raise ValueError(f"No legal contraction remains at {len(mesh.faces)} faces before target {target}.")

            # The chosen stars do not share changed faces, so their independent
            # contractions can be applied together without invalidating the face tests.
            mesh, kept, merged = plumbing.contract(mesh, chosen_edges)
            vertices = kept * vertices + merged * optima                        # Point-[vertices]
            quadrics = kept * quadrics + merged * edge_quadrics                 # (Plane+ <- Point-)[vertices]
        yield mesh, vertices
