"""Mechanical work pairs a force applied at a handle with the motion of its crank.

Projective geometric algebra models handles and directions as points, intrinsic antivectors, and a crank's rate of
turning as an extrinsic bivector. Joining a handle to its applied force gives the forque that pairs with this rate,
`forque & rate`.
"""

from __future__ import annotations

import numpy as np

from numga import NumpyContext
from numga.algebras import PGA2D

ga = PGA2D
mv = NumpyContext(ga).multivector
Point = ga.gatype.antivector()                                                  # Antivector-
Rate = ga.gatype.bivector()                                                     # Bivector+
Scalar = ga.gatype.scalar()                                                     # Scalar+


# --- math -----------------------------------------------------------------------------
def turn(handles: Point, forces: Point, rates: Rate, times: np.ndarray) -> tuple[Point, Point, Point, Scalar]:
    """The moving handles, their velocities, applied forces and powers over one turn."""
    poses = (rates[None, :] * (-times[:, None] / 2)).exp()                      # [steps, cranks] Motor+
    handles = poses >> handles                                                  # [steps, cranks] Point-
    forces = poses >> forces                                                    # [steps, cranks] Direction-

    # The handle's velocity follows directly from its rotation rate.
    velocities = handles.commutator(rates)                                      # [steps, cranks] Direction-

    # The force's point of application determines its forque on the crank.
    forques = handles & forces                                                  # [steps, cranks] Forque-
    powers = forques & rates                                                    # [steps, cranks] Scalar+
    return handles, velocities, forces, powers
