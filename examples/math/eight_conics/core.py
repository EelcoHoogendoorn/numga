"""Penrose's eight-conic theorem, in the projective geometric algebra of space.

Quadrics on the corners of a cube touch their neighbours in double contact: two points with common
tangents. Each quadric maps a point to its polar plane, and adding the dyad of a plane changes it everywhere
but on that plane, so double contact is an addition of maps. Three compatible faces determine the eighth
quadric in three ways, by one more contact dyad from each of its neighbours, and the three are one map.
"""

from __future__ import annotations

import numpy as np

from numga import NumpyContext, stack
from numga.algebras import PGA2D as ga

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                         # Scalar+
Plane = ga.gatype.vector()                          # Vector+
Point = ga.gatype.antivector()                      # Antivector-
Quadric = ga.gatype((Plane, Point))

# The three faces opposite the first neighbours use the other two contact planes.
FORWARD = np.array([1, 2, 0])
BACKWARD = np.array([2, 0, 1])
EDGES = np.array([[0, 1], [0, 2], [0, 4], [1, 3], [2, 3], [1, 5],
                  [4, 5], [2, 6], [4, 6], [3, 7], [5, 7], [6, 7]])


# --- math -----------------------------------------------------------------------------
def configuration(base: Quadric, contact_planes: Plane, couplings: np.ndarray) -> tuple[Quadric, Plane, Quadric]:
    """The cube completed from a quadric, three weighted contact planes and three face couplings: its eight quadrics,
    its twelve contact planes, and the eighth quadric's three independent completions, on the last axis."""
    double_planes = contact_planes * (contact_planes & Point)               # [3] Plane+ <- Point-
    neighbours = base + double_planes                                       # [3] Plane+ <- Point-
    left = contact_planes[FORWARD]                                          # [3] Plane+
    right = contact_planes[BACKWARD]                                        # [3] Plane+
    face_scales = 1 - couplings ** 2

    # Each face quadric shares one contact plane with each of two neighbours.
    cross_terms = left * (right & Point) + right * (left & Point)             # [3] Plane+ <- Point-
    faces = (base * face_scales + double_planes[FORWARD] + double_planes[BACKWARD]
             + cross_terms * couplings)                                     # [..., 3] Plane+ <- Point-

    # The three closing contact planes determine the same eighth quadric independently.
    closing_scale = 1 - np.sum(couplings ** 2, axis=-1) - 2 * np.prod(couplings, axis=-1)
    closing_planes = (contact_planes * face_scales
                      + left * (couplings[..., BACKWARD] + couplings * couplings[..., FORWARD])
                      + right * (couplings[..., FORWARD] + couplings * couplings[..., BACKWARD]))
    completions = ((faces * closing_scale[..., None]
                    + closing_planes * (closing_planes & Point)) / face_scales)  # [..., 3] Plane+ <- Point-

    # Binary vertex labels make each cube edge change one of the three choices.
    shape = completions.shape[:-1]
    quadrics = stack((base.broadcast_to(shape), neighbours[0].broadcast_to(shape),
                    neighbours[1].broadcast_to(shape), faces[..., 2],
                    neighbours[2].broadcast_to(shape), faces[..., 1], faces[..., 0],
                    completions[..., 0]), axis=-1)                           # [..., 8] Plane+ <- Point-
    left_planes = left * couplings + right                                  # [..., 3] Plane+
    right_planes = left + right * couplings                                 # [..., 3] Plane+
    edge_planes = stack((contact_planes[0].broadcast_to(shape), contact_planes[1].broadcast_to(shape),
                         contact_planes[2].broadcast_to(shape), left_planes[..., 2], right_planes[..., 2],
                         right_planes[..., 1], left_planes[..., 1], left_planes[..., 0],
                         right_planes[..., 0], closing_planes[..., 2],
                         closing_planes[..., 1], closing_planes[..., 0]), axis=-1)  # [..., 12] Plane+
    return quadrics, edge_planes, completions


# --- plumbing -------------------------------------------------------------------------
def intersections(quadrics: Quadric, planes: Plane) -> Point:
    """The two real crossings of each quadric with its plane, on the last batch axis."""
    origin = mv.w.dual()
    planes = planes.normalized()
    # Reflection gives the foot of the perpendicular from the chart origin.
    anchor = (origin + (planes >> origin)) / 2                              # [...] Point-
    direction = (planes | mv.xy).dual()                                     # [...] Point-
    quadratic = quadrics(direction) & direction                             # [...] Scalar+
    linear = quadrics(anchor) & direction                                   # [...] Scalar+
    constant = quadrics(anchor) & anchor                                    # [...] Scalar+
    root_signs = np.array([-1, 1])
    distances = (-linear[..., None]
                 + (linear.squared() - quadratic * constant).square_root()[..., None]
                 * root_signs) / quadratic[..., None]                       # [..., 2] Scalar+
    return anchor[..., None] + direction[..., None] * distances               # [..., 2] Point-
