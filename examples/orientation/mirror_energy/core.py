"""Kinetic energy stays positive when a moving mass distribution is reflected.

Spherical geometry in the algebra x+y+z+ models positions as intrinsic antivectors and rates of turning as extrinsic
bivectors. Joining a position to its mass-weighted velocity gives the momentum, intrinsic, and the energy is half the
momentum paired with the rate, `momentum & rate`.
"""

from __future__ import annotations

import numpy as np

from numga import NumpyContext, stack
from numga.algebras import Spherical3D

ga = Spherical3D
mv = NumpyContext(ga).multivector
Point = ga.gatype.antivector()                                                  # Antivector-
Plane = ga.gatype.vector()                                                      # Vector+
Rate = ga.gatype.bivector()                                                     # Bivector+
Scalar = ga.gatype.scalar()                                                     # Scalar+


# --- math -----------------------------------------------------------------------------
def energies(points: Point, masses: np.ndarray, rate: Rate, mirror: Plane) -> Scalar:
    """The kinetic energy of the turning masses, of their reflection applied to the inertia's input and output, of the
    inertia rebuilt from the reflected masses, and of the plain sandwich that reflects every element alike, in that
    order."""
    # Inertia maps the body's rate of motion to its momentum.
    inertia = points.regressive(points.commutator(Rate) * masses).sum(axis=0)   # [] Momentum- <- Rate+
    energy = (inertia(rate) & rate) / 2                                         # [] Scalar+

    # Reflect the map on its input and its output, along with the actual motion rate.
    reflected_rate = mirror >> rate                                             # [] Rate+
    reflected_inertia = mirror >> inertia(mirror << Rate)                       # [] Momentum- <- Rate+
    reflected_energy = (reflected_inertia(reflected_rate) & reflected_rate) / 2  # [] Scalar+

    # Constructing from reflected masses gives the same inertia and energy.
    reflected_points = mirror >> points                                         # [masses] Point-
    rebuilt_inertia = reflected_points.regressive(reflected_points.commutator(Rate) * masses).sum(axis=0)  # [] Momentum- <- Rate+
    rebuilt_energy = (rebuilt_inertia(reflected_rate) & reflected_rate) / 2     # [] Scalar+

    # Uniform grade-involuted conjugation gives momentum the wrong sign.
    sandwich_momentum = mirror * inertia(rate).involute() * mirror.inverse()    # [] Momentum-
    sandwich_rate = mirror * rate.involute() * mirror.inverse()                 # [] Rate+
    sandwich_energy = (sandwich_momentum & sandwich_rate) / 2                   # [] Scalar+
    return stack([energy, reflected_energy, rebuilt_energy, sandwich_energy])


def motion(points: Point, rate: Rate, mirror: Plane, times: np.ndarray) -> tuple[Point, Point]:
    """The positions of the masses turned at the rate for each of the times, beside their mirror image, and their
    velocities, along the axes of times, the two cases and the masses."""
    turns = (rate * (-times[:, None] / 2)).exp()                                # [times, 1] Rotor+
    positions = turns >> points                                                 # [times, masses] Point-
    # A point turning at a rate moves along its commutator with the rate.
    velocities = positions.commutator(rate)                                     # [times, masses] Velocity-
    return stack([positions, mirror >> positions], axis=1), stack([velocities, mirror >> velocities], axis=1)
