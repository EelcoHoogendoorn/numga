"""Garland–Heckbert Quadric Error Metrics (QEM) in PGA3D.

In mesh simplification and surface decimation, each triangle face defines a
supporting plane. A point's squared distance to a plane is a rank-one dyad.
Summing these dyads across incident faces yields the quadric error metric (QEM):
an extensor mapping points to polar planes.

Edge contraction combines the quadrics of its endpoints by extensor addition:
Q_edge = Q_a + Q_b. The optimal collapsed vertex position has vanishing spatial
gradient, meaning its polar plane is the plane at infinity: Q_edge(X) ∝ ∞.
Solving Q_edge.lstsq(mv.w) yields the optimal vertex in a single linear solve.

This module contains the mathematics alone: GATypes and the geometric narrative
in one coherent scope.
"""

from __future__ import annotations

from numga import NumpyContext
from numga.algebras import PGA3D

# --- algebra and types -----------------------------------------------------
ga = PGA3D
ctx = NumpyContext(ga)
mv = ctx.multivector

Point = ga.gatype.antivector()
Plane = ga.gatype.vector()
Quadric = ga.gatype((Plane, Point))


# --- math ------------------------------------------------------------------
def edge_collapse(
    planes_a: Plane,
    planes_b: Plane,
) -> tuple[Quadric, Quadric, Quadric, Point]:
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
    # is the plane at infinity: Q(X) ∝ mv.w.
    v_edge: Point = q_edge.lstsq(mv.w).normalized()

    return qa, qb, q_edge, v_edge
