"""Garland–Heckbert Quadric Error Metrics (QEM) in PGA3D.

In mesh simplification and surface decimation, each triangle face defines a
supporting plane. A point's squared distance to a plane is a rank-one dyad.
Summing these dyads across incident faces yields the quadric error metric (QEM):
an extensor mapping points to polar planes.

Edge contraction combines the quadrics of its endpoints by extensor addition. The optimal
collapsed vertex has vanishing spatial gradient at unit homogeneous weight. Solving for a
direction from the edge midpoint also gives a definite answer on a flat patch or a crease.
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

Point = ga.gatype.antivector()
Plane = ga.gatype.vector()
Quadric = ga.gatype((Plane, Point))
Spatial = ga.gatype.from_blades("x y z")
Direction = Spatial.dual().output_gatype


# --- math -----------------------------------------------------------------------------
def collapse_position(quadric: Quadric, midpoint: Point) -> Point:
    """The unit-weight minimum, nearest the midpoint when the quadric has flat directions."""
    # A direction changes position without changing homogeneous weight. Only the spatial
    # part of the polar plane must vanish; its constant part is the residual minimum error.
    correction = quadric(Direction).cast(Spatial).lstsq(-quadric(midpoint).cast(Spatial))
    return midpoint + correction


def simplify(vertices: Point, faces: np.ndarray, targets: np.ndarray) -> Iterator[tuple[Point, np.ndarray]]:
    """A closed triangle mesh, then its QEM contractions at descending face-count targets.

    The initial face planes remain the reference surface: every merged vertex retains the
    sum of its original vertices' quadrics. Each pass contracts separate neighbourhoods,
    choosing the least errors first and preserving the topology and the sense of each face.
    """
    corners = vertices[faces]
    planes = (corners[:, 0] & corners[:, 1] & corners[:, 2]).normalized()
    face_quadrics = planes * (planes & Point)
    quadrics = (face_quadrics[0] * 0).broadcast_to((len(vertices),))
    quadrics = quadrics.at[faces.reshape(-1)].add(face_quadrics[:, None].broadcast_to((len(faces), 3)).reshape(-1))
    yield vertices, faces

    for target in targets:
        while len(faces) > target:
            topology = plumbing.Topology(faces, len(vertices))
            first, second = topology.edges.T
            edge_quadrics = quadrics[first] + quadrics[second]
            midpoints = (vertices[first] + vertices[second]) / 2
            optima = collapse_position(edge_quadrics, midpoints)
            errors = edge_quadrics(optima) & optima

            # Test every surviving triangle in an edge's neighbourhood with that edge
            # contracted. Its plane must keep its sense and a nonzero area.
            before = vertices[faces[topology.trial_faces]]
            after = before.at[np.arange(len(before)), topology.trial_corners].set(optima[topology.trial_edges])
            old_planes = before[:, 0] & before[:, 1] & before[:, 2]
            new_planes = after[:, 0] & after[:, 1] & after[:, 2]
            consistent = (old_planes.scalar_product(new_planes) > 0) & (new_planes.squared() > old_planes.squared() * 1e-12)
            selected = plumbing.select_edges(topology, errors, consistent, max(1, (len(faces) - target) // 2))
            if not len(selected):
                raise ValueError(f"No legal contraction remains at {len(faces)} faces before target {target}.")

            # The selected stars do not share changed faces, so their independent
            # contractions can be applied together without invalidating the face tests.
            vertices = vertices.at[first[selected]].set(optima[selected])
            quadrics = quadrics.at[first[selected]].set(edge_quadrics[selected])
            retained, faces = plumbing.contract(topology, selected)
            vertices, quadrics = vertices[retained], quadrics[retained]
        yield vertices, faces
