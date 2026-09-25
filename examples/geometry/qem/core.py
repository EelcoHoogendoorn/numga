"""Garland–Heckbert Quadric Error Metrics (QEM) in PGA3D.

In mesh simplification and surface decimation, each triangle face defines a
supporting plane. A point's squared distance to a plane is a rank-one dyad.
Summing these dyads across incident faces yields the quadric error metric (QEM):
an extensor mapping points to polar planes.

Edge contraction combines the quadrics of its endpoints by extensor addition:
`q_edge = qa + qb`. The optimal collapsed vertex position has vanishing spatial
gradient, meaning its polar plane is the plane at infinity: `q_edge(v_edge)` is a
multiple of `mv.w`. Solving `q_edge.lstsq(mv.w)` yields the optimal vertex in a single
linear solve.
"""

from __future__ import annotations

import numpy as np

from numga import NumpyContext
from numga.algebras import PGA3D

# --- algebra and types -----------------------------------------------------
ga = PGA3D
ctx = NumpyContext(ga)
mv = ctx.multivector

Point = ga.gatype.antivector()
Plane = ga.gatype.vector()
Quadric = ga.gatype((Plane, Point))


def point(coords: np.ndarray) -> Point:
    """Finite points at (..., 3) coordinates: the dual of the homogeneous vector."""
    return (mv("x y z", coords) + mv.w).dual()


def direction(coords: np.ndarray) -> Point:
    """Ideal points: the directions (..., 3), the dual of a weightless vector."""
    return mv("x y z", coords).dual()


# --- plumbing ----------------------------------------------------------------
def ridge_patch() -> tuple[Point, np.ndarray]:
    """A surface patch with a sharp ridge ending in an apex: six vertices and six triangles."""
    vertices = point(np.array([
        [ 0.35,  0.0,   0.25],  # 0: vertex a (sharp corner apex)
        [-0.35,  0.0,   0.25],  # 1: vertex b (crease continuation)
        [ 0.0,   0.55, -0.15],  # 2: left base vertex
        [ 0.0,  -0.55, -0.15],  # 3: right base vertex
        [ 0.75,  0.0,  -0.15],  # 4: front tip (steep corner at a)
        [-0.85,  0.0,  -0.15],  # 5: back tip (gentle ramp at b)
    ]))
    faces = np.array([
        [0, 1, 2],  # 0: left flank (shared by edge a-b)
        [1, 0, 3],  # 1: right flank (shared by edge a-b)
        [0, 2, 4],  # 2: corner front-left (incident to a only)
        [0, 4, 3],  # 3: corner front-right (incident to a only)
        [1, 5, 2],  # 4: ramp back-left (incident to b only)
        [1, 3, 5],  # 5: ramp back-right (incident to b only)
    ])
    return vertices, faces


# --- math ------------------------------------------------------------------
def edge_collapse(
    planes_a: Plane,
    planes_b: Plane,
):
    """Contract a mesh edge (a, b) by summing incident face plane quadrics.

    planes_a and planes_b are the supporting planes of the mesh faces incident
    to the edge's two endpoint vertices, a and b. In a 2-manifold mesh, the
    faces flanking the edge contain both endpoints and appear in both sets.
    """
    # A plane P measures scalar distance to a point X as P & X. Leaving the point slot
    # open yields a rank-1 dyad P * (P & Point): Plane <- Point.
    # Summing incident plane dyads at each vertex yields its error quadric:
    qa: Quadric = (planes_a * (planes_a & Point)).sum(axis=0)
    qb: Quadric = (planes_b * (planes_b & Point)).sum(axis=0)

    # Edge contraction combines quadrics across endpoints by extensor addition:
    q_edge: Quadric = qa + qb

    # The optimal vertex has vanishing spatial gradient, so its polar plane
    # is the plane at infinity: q_edge(v_edge) is a multiple of mv.w.
    v_edge: Point = q_edge.lstsq(mv.w).normalized()

    return qa, qb, q_edge, v_edge
