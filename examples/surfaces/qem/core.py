"""Garland–Heckbert quadric error metrics simplify a triangle mesh by collapsing edges to vertices that minimize squared distances to incident face planes.

In 3D projective geometric algebra, a plane is modelled as a vector and a point is modelled as an antivector (`Point`), so their regressive pairing `plane & point` gives the point's signed distance from a unit plane. A plane's dyad `plane * (plane & Point)` maps points to polar planes, and pairing that polar plane back with the point evaluates the squared distance. The error quadric of a vertex is the sum of its incident triangles' plane dyads, and collapsing an edge adds the quadrics of its two endpoints. Solving for a spatial direction offset `Direction` from the edge's midpoint finds the minimum at unit homogeneous weight and selects the minimizer nearest the midpoint when a flat patch or crease leaves directions free, following the quadric error metric of Garland and Heckbert, https://www.cs.cmu.edu/~garland/quadrics/.

In homogeneous matrix notation each unit plane reads as a four-vector $\\mathbf p = (n_x, n_y, n_z, d)^{\\mathsf T}$, its plane dyad reads as the outer product $Q = \\mathbf p\\mathbf p^{\\mathsf T}$, and the squared distance at $\\mathbf x = (x, y, z, 1)^{\\mathsf T}$ reads as the quadratic form $\\mathbf x^{\\mathsf T} Q\\,\\mathbf x$.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import NumpyContext
from numga.algebras import PGA3D

from examples.surfaces.qem import plumbing

ga = PGA3D
ctx = NumpyContext(ga)
mv = ctx.multivector

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


def simplify(vertices: Point, faces: np.ndarray, targets: np.ndarray) -> Iterator[tuple[Point, np.ndarray]]:
    """Yield a closed triangle mesh and its QEM contractions at descending face-count targets.

    The initial face planes remain the reference surface: every merged vertex retains the
    sum of its initial vertices' quadrics. Each pass contracts separate neighbourhoods,
    choosing the least errors first and preserving the topology and the sense of each face.
    """
    corners = vertices[faces]                                                   # [faces, 3] Point-
    planes = (corners[:, 0] & corners[:, 1] & corners[:, 2]).normalized()       # [faces] Plane-
    face_quadrics = planes * (planes & Point)                                   # [faces] Plane+ <- Point-
    quadrics = (face_quadrics[0] * 0).broadcast_to((len(vertices),))            # [vertices] Plane+ <- Point-
    quadrics = quadrics.at[faces.reshape(-1)].add(face_quadrics[:, None].broadcast_to((len(faces), 3)).reshape(-1))
    yield vertices, faces

    for target in targets:
        while len(faces) > target:
            topology = plumbing.Topology(faces, len(vertices))
            first, second = topology.edges.T
            edge_quadrics = quadrics[first] + quadrics[second]                  # [edges] Plane+ <- Point-
            midpoints = (vertices[first] + vertices[second]) / 2                # [edges] Point-
            optima = collapse_position(edge_quadrics, midpoints)                # [edges] Point-
            errors = edge_quadrics(optima) & optima                             # [edges] Scalar+

            # Test every surviving triangle in an edge's neighbourhood with that edge
            # contracted. Its plane must keep its sense and a nonzero area.
            old_corners = vertices[faces[topology.trial_faces]]                 # [trials, 3] Point-
            new_corners = old_corners.at[np.arange(len(old_corners)), topology.trial_corners].set(optima[topology.trial_edges])
            old_planes = old_corners[:, 0] & old_corners[:, 1] & old_corners[:, 2]  # [trials] Plane-
            new_planes = new_corners[:, 0] & new_corners[:, 1] & new_corners[:, 2]  # [trials] Plane-
            valid_trials = (old_planes.scalar_product(new_planes) > 0) & (new_planes.squared() > old_planes.squared() * 1e-12)
            chosen_edges = plumbing.select_edges(topology, errors, valid_trials, max(1, (len(faces) - target) // 2))
            if not len(chosen_edges):
                raise ValueError(f"No legal contraction remains at {len(faces)} faces before target {target}.")

            # The chosen stars do not share changed faces, so their independent
            # contractions can be applied together without invalidating the face tests.
            vertices = vertices.at[first[chosen_edges]].set(optima[chosen_edges])
            quadrics = quadrics.at[first[chosen_edges]].set(edge_quadrics[chosen_edges])
            kept_vertices, faces = plumbing.contract(topology, chosen_edges)
            vertices, quadrics = vertices[kept_vertices], quadrics[kept_vertices]
        yield vertices, faces
