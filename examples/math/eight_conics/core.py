"""Penrose's eight-conic theorem, in the projective geometric algebra of the plane.

Conics on the corners of a cube touch their neighbours in double contact: two points with common
tangents. Each conic maps a point to its polar line, and adding the dyad of a line changes it everywhere
but on that line, so double contact is an addition of maps. Three compatible faces determine the eighth
conic in three ways, by one more contact dyad from each of its neighbours, and the three are one map.
"""

from __future__ import annotations

import numpy as np

from numga import NumpyContext, stack
from numga.algebras import PGA2D as ga

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()
Plane = ga.gatype.vector()
Point = ga.gatype.antivector()
Conic = ga.gatype((Plane, Point))

# The three faces opposite the first neighbours use the other two chords.
FORWARD = np.array([1, 2, 0])
BACKWARD = np.array([2, 0, 1])
EDGES = np.array([[0, 1], [0, 2], [0, 4], [1, 3], [2, 3], [1, 5],
                  [4, 5], [2, 6], [4, 6], [3, 7], [5, 7], [6, 7]])


# --- math -----------------------------------------------------------------------------
def configuration(base: Conic, chords: Plane, couplings: np.ndarray) -> tuple[Conic, Plane, Conic]:
    """The cube completed from a conic, three weighted chords and three face couplings: its eight conics,
    its twelve contact chords, and the eighth conic's three independent completions, on the last axis."""
    double_lines = chords * (chords & Point)                                  # [3] Plane <- Point
    neighbours = base + double_lines                                         # [3] Plane <- Point
    left = chords[FORWARD]                                                   # [3] Plane
    right = chords[BACKWARD]                                                 # [3] Plane
    face_scales = 1 - couplings ** 2

    # Each face conic shares one contact chord with each of two neighbours.
    cross_terms = left * (right & Point) + right * (left & Point)             # [3] Plane <- Point
    faces = (base * face_scales + double_lines[FORWARD] + double_lines[BACKWARD]
             + cross_terms * couplings)                                     # [..., 3] Plane <- Point

    # The three closing chords determine the same eighth conic independently.
    closing_scale = 1 - np.sum(couplings ** 2, axis=-1) - 2 * np.prod(couplings, axis=-1)
    closing_chords = (chords * face_scales
                      + left * (couplings[..., BACKWARD] + couplings * couplings[..., FORWARD])
                      + right * (couplings[..., FORWARD] + couplings * couplings[..., BACKWARD]))
    completions = ((faces * closing_scale[..., None]
                    + closing_chords * (closing_chords & Point)) / face_scales)  # [..., 3] Plane <- Point

    # Binary vertex labels make each cube edge change one of the three choices.
    shape = completions.shape[:-1]
    conics = stack((base.broadcast_to(shape), neighbours[0].broadcast_to(shape),
                    neighbours[1].broadcast_to(shape), faces[..., 2],
                    neighbours[2].broadcast_to(shape), faces[..., 1], faces[..., 0],
                    completions[..., 0]), axis=-1)                           # [..., 8] Plane <- Point
    left_chords = left * couplings + right                                  # [..., 3] Plane
    right_chords = left + right * couplings                                 # [..., 3] Plane
    edge_chords = stack((chords[0].broadcast_to(shape), chords[1].broadcast_to(shape),
                         chords[2].broadcast_to(shape), left_chords[..., 2], right_chords[..., 2],
                         right_chords[..., 1], left_chords[..., 1], left_chords[..., 0],
                         right_chords[..., 0], closing_chords[..., 2],
                         closing_chords[..., 1], closing_chords[..., 0]), axis=-1)  # [..., 12] Plane
    return conics, edge_chords, completions


# --- plumbing -------------------------------------------------------------------------
def intersections(conics: Conic, lines: Plane) -> Point:
    """The two real crossings of each conic with its line, on the last batch axis."""
    origin = mv.w.dual()
    lines = lines.normalized()
    # Reflection gives the foot of the perpendicular from the chart origin.
    anchor = (origin + (lines >> origin)) / 2                                # [...] Point
    direction = (lines | mv.xy).dual()                                      # [...] Point
    quadratic = conics(direction) & direction                               # [...] Scalar
    linear = conics(anchor) & direction                                     # [...] Scalar
    constant = conics(anchor) & anchor                                      # [...] Scalar
    root_signs = np.array([-1, 1])
    distances = (-linear[..., None]
                 + (linear.squared() - quadratic * constant).square_root()[..., None]
                 * root_signs) / quadratic[..., None]                       # [..., 2] Scalar
    return anchor[..., None] + direction[..., None] * distances               # [..., 2] Point
