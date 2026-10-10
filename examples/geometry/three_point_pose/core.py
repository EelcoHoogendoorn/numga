"""Recover a rigid pose by successively matching a point, a line and a plane.

The construction follows the motor reconstruction example of ganja.js by Steven De Keninck,
https://github.com/enkimute/ganja.js/blob/master/examples/example_chapter11_motor_reconstruction.html.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import NumpyContext, stack
from numga.algebras import PGA3D

ga = PGA3D
mv = NumpyContext(ga).multivector
Point = ga.gatype.antivector()
Line = ga.gatype.bivector()
Plane = ga.gatype.vector()
Motor = ga.gatype.rotor()


# --- math -----------------------------------------------------------------------------
def reconstruct(source: Point, target: Point) -> Motor:
    """Three incremental motors for congruent, ordered, noncollinear marker triples, along a leading stage axis.

    Each alignment uses the short motor branch; opposing oriented elements must
    not occur at an intermediate step.
    """
    # Two point reflections translate by twice the required displacement.
    translation = (target[..., 0] / source[..., 0]).square_root()

    # Turn the joining line while keeping the matched first point fixed.
    target_line = (target[..., 0] & target[..., 1]).normalized()
    current_line = (target[..., 0] & (translation >> source[..., 1])).normalized()
    turn = (target_line / current_line).square_root()
    placed = turn * translation

    # Only rotation about that line remains. Match the marker planes to remove it.
    target_plane = (target_line & target[..., 2]).normalized()
    current_plane = (target_line & (placed >> source[..., 2])).normalized()
    roll = (target_plane / current_plane).square_root()
    return stack([translation, turn, roll])


def placements(source: Point, increments: Motor) -> Iterator[Point]:
    """The markers after each alignment."""
    for increment in increments:
        source = increment >> source
        yield source


def alignments(source: Point, increments: Motor, fractions: np.ndarray) -> Iterator[Point]:
    """Carry markers through each alignment, retaining the constraints already satisfied."""
    placed = source
    yield placed
    for increment in increments:
        start = placed
        for fraction in fractions:
            # A fraction of the motor's logarithm moves at a constant rate along its screw.
            placed = (increment.log() * fraction).exp() >> start
            yield placed


# --- plumbing -------------------------------------------------------------------------
def point(coords: np.ndarray) -> Point:
    """Finite points from construction coordinates with shape (..., 3)."""
    return (mv("x y z", coords) + mv.w).dual()
