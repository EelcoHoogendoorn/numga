"""Mechanical work pairs an applied force's line of action with the turning rate of a crank.

In planar projective geometric algebra, handle positions are modelled as points and force directions as ideal points,
both intrinsic antivectors, while a crank's turning rate is modelled as an extrinsic bivector. Joining a handle with
its applied force gives the intrinsic forque `handle & force`, and pairing the rate with that forque, `rate & forque`,
gives the mechanical power as an extrinsic scalar that reflection leaves unchanged.
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
def crank_motion(handles: Point, forces: Point, rates: Rate, times: np.ndarray) -> tuple[Point, Point, Point, Scalar]:
    """Turning `handles` and `forces` (`[n_cranks] Point-`) at `rates` (`[n_cranks] Rate+`) across `times`
    (`[n_steps]`) produces `[n_steps, n_cranks]` displaced handles, handle velocities, turned forces, and mechanical
    powers."""
    poses = (rates[None, :] * (-times[:, None] / 2)).exp()                      # [...] Motor+
    turned_handles = poses >> handles                                           # [...] Point-
    turned_forces = poses >> forces                                             # [...] Point-

    # Each handle's velocity along its circular path is its commutator with the turning rate.
    velocities = turned_handles.commutator(rates)                               # [...] Point-

    # Joining the handle with its applied force forms the intrinsic forque paired against the rate.
    forques = turned_handles & turned_forces                                    # [...] Forque-
    powers = rates & forques                                                    # [...] Scalar+
    return turned_handles, velocities, turned_forces, powers
