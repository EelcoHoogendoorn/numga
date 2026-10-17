"""Loop subdivision of weighted points of projective space.

A Loop step splits every triangle of a closed mesh in four. Each vertex moves to an average of itself and
its neighbours, and each edge gets a new vertex, an average of its two ends and the two vertices across
from it. The averages are a map on the vertex sites. Here the sites hold points of projective space, each
with its own weight, and the step adds and scales the points as they are: a heavy vertex pulls the surface
towards itself, a light one lets it go. A projective map acts on the points at the sites and the step on
the sites themselves, so the two commute, whatever the weights; averaging positions instead commutes with
affine maps only.
"""

from __future__ import annotations

from collections.abc import Iterator
from functools import reduce

import numpy as np

from numga import NumpyContext
from numga.algebras import PGA3D as ga
from numga.sparse import SparseExtensor
from examples import instantiate

ctx = NumpyContext(ga)
mv = ctx.multivector
Mesh = instantiate("examples.mesh", ga).Mesh
Scalar = ga.gatype.scalar()
Plane = ga.gatype.vector()
Point = ga.gatype.antivector()
PointMap = ga.gatype((Point, Point))                                           # Point <- Point


# --- math -----------------------------------------------------------------------------
def refinements(mesh: Mesh, pinned: np.ndarray, depth: int) -> Iterator[tuple[SparseExtensor, Mesh]]:
    """The map of each of so many Loop steps from a mesh, with the finer mesh it gives."""
    for _ in range(depth):
        step = mesh.loop(pinned)                                               # [finer, coarser] Scalar
        mesh, pinned = Mesh(step * mesh.vertices, mesh.split_faces), mesh.split_pins(pinned)
        yield step, mesh


def subdivision(mesh: Mesh, depth: int) -> tuple[SparseExtensor, Mesh]:
    """So many Loop steps from a closed mesh, nothing pinned, composed into one map from its vertex sites to
    those of the refined mesh, and the refined mesh."""
    steps, meshes = zip(*refinements(mesh, np.zeros(mesh.vertices.batch().shape[-1], dtype=bool), depth))
    return reduce(lambda total, step: step * total, steps), meshes[-1]       # [refined, vertices] Scalar


def unit(points: Point) -> Point:
    """The points divided by their weights: their positions, at unit weight."""
    return points / (mv.w & points)


def collineation(centre: Point, axis: Plane, strength: float) -> PointMap:
    """The central collineation of a centre and an axis plane: each point moves along its line through the
    centre, by the centre times its pairing with the axis; points on the axis stay where they are."""
    return Point + centre * (axis & Point) * strength                         # [] Point <- Point
