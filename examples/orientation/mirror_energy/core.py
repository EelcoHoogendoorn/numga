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
    inertia: Inertia = (points & points.commutator(Rate) * masses).sum(axis=0)
    energy: Scalar = (inertia(rate) & rate) / 2

    # Reflect the map on both slots, along with the actual motion rate.
    reflected_rate: Rate = mirror >> rate
    reflected_inertia: Inertia = mirror >> inertia(mirror << Rate)
    reflected_energy: Scalar = (reflected_inertia(reflected_rate) & reflected_rate) / 2

    # Constructing from reflected masses gives the same inertia and energy.
    reflected_points: Point = mirror >> points
    rebuilt: Inertia = (reflected_points & reflected_points.commutator(Rate) * masses).sum(axis=0)
    rebuilt_energy: Scalar = (rebuilt(reflected_rate) & reflected_rate) / 2

    # Uniform grade-involuted conjugation gives momentum the wrong sign.
    untyped_momentum: Momentum = mirror * inertia(rate).involute() * mirror.inverse()
    untyped_rate: Rate = mirror * rate.involute() * mirror.inverse()
    untyped_energy: Scalar = (untyped_momentum & untyped_rate) / 2
    return stack([energy, reflected_energy, rebuilt_energy, untyped_energy])


def motion(points: Point, rate: Rate, mirror: Plane, times: np.ndarray) -> tuple[Point, Point]:
    """Sample a prescribed rotation and its mirror, with the tangent velocities."""
    positions: Point = (-rate * times[:, None] / 2).exp() >> points
    velocities: Point = positions.commutator(rate)
    return stack([positions, mirror >> positions], axis=1), stack([velocities, mirror >> velocities], axis=1)
