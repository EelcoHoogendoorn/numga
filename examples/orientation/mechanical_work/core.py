"""Mechanical work pairs a force applied at a handle with the motion of its crank.

Projective geometric algebra holds positions and directions as points, and a motion rate as
a bivector. Joining a handle to its applied force gives the load that pairs with this rate.
"""

from __future__ import annotations

import numpy as np

from numga import NumpyContext
from numga.algebras import PGA2D

ga = PGA2D
mv = NumpyContext(ga).multivector
Point = ga.gatype.antivector()
Plane = ga.gatype.vector()
Rate = ga.gatype.bivector()
Load = ga.gatype.antibivector()
Scalar = ga.gatype.scalar()


# --- math -----------------------------------------------------------------------------
def turn(handles: Point, forces: Point, rates: Rate, times: np.ndarray) -> tuple[Point, Point, Point, Scalar]:
    """The moving handles, their velocities, applied forces and powers over one turn."""
    poses = (rates[None, :] * (-times[:, None] / 2)).exp()
    handles = poses >> handles
    forces = poses >> forces

    # The handle's velocity follows directly from its rotation rate.
    velocities: Point = handles.commutator(rates)

    # The force's point of application determines its load on the crank.
    loads: Load = handles & forces
    powers: Scalar = loads & rates
    return handles, velocities, forces, powers
