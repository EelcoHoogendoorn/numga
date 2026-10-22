"""Kinetic energy stays positive when a moving mass distribution is reflected.

Spherical geometry in the algebra x+y+z+ represents positions by antivectors and
motion rates by bivectors. Joining a position to its mass-weighted velocity gives momentum.
"""

from __future__ import annotations

import numpy as np

from numga import NumpyContext, stack
from numga.algebras import Spherical3D

ga = Spherical3D
mv = NumpyContext(ga).multivector
Point = ga.gatype.antivector()
Plane = ga.gatype.vector()
Rate = ga.gatype.bivector()
Momentum = ga.gatype.antibivector()
Inertia = ga.gatype((Momentum, Rate))
Scalar = ga.gatype.scalar()


# --- math -----------------------------------------------------------------------------
def energies(points: Point, masses: np.ndarray, rate: Rate, mirror: Plane) -> Scalar:
    """Energy before reflection, after reflection, after rebuilding, and without orientation."""
    # Inertia maps the body's rate of motion to its momentum.
    inertia = (points & points.commutator(Rate) * masses).sum(axis=0)           # Inertia
    energy = (inertia(rate) & rate) / 2                                         # Scalar

    # Reflect the map on its input and its output, along with the actual motion rate.
    reflected_rate = mirror >> rate                                             # Rate
    reflected_inertia = mirror >> inertia(mirror << Rate)                       # Inertia
    reflected_energy = (reflected_inertia(reflected_rate) & reflected_rate) / 2  # Scalar

    # Constructing from reflected masses gives the same inertia and energy.
    reflected_points = mirror >> points                                         # Point
    rebuilt = (reflected_points & reflected_points.commutator(Rate) * masses).sum(axis=0)  # Inertia
    rebuilt_energy = (rebuilt(reflected_rate) & reflected_rate) / 2             # Scalar

    # Uniform grade-involuted conjugation gives momentum the wrong sign.
    untyped_momentum = mirror * inertia(rate).involute() * mirror.inverse()     # Momentum
    untyped_rate = mirror * rate.involute() * mirror.inverse()                  # Rate
    untyped_energy = (untyped_momentum & untyped_rate) / 2                      # Scalar
    return stack([energy, reflected_energy, rebuilt_energy, untyped_energy])


def motion(points: Point, rate: Rate, mirror: Plane, times: np.ndarray) -> tuple[Point, Point]:
    """Sample a prescribed rotation and its mirror, with the tangent velocities."""
    positions = (-rate * times[:, None] / 2).exp() >> points                    # Point
    velocities = positions.commutator(rate)                                     # Point
    return stack([positions, mirror >> positions], axis=1), stack([velocities, mirror >> velocities], axis=1)
