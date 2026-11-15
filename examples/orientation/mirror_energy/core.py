"""Rotational kinetic energy remains positive when a spherical mass distribution and its motion are reflected.

On the unit sphere in three-dimensional geometric algebra, mass positions are modelled as intrinsic antivectors and
turning rates as extrinsic bivectors. Joining each weighted position with its velocity forms the intrinsic angular
momentum, and half the pairing `rate & momentum` evaluates the kinetic energy as an extrinsic scalar.
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
    """Half the pairing `rate & momentum` gives `[4] Scalar+` kinetic energies for `points` (`[n_masses] Point-`): the
    original body, the reflected inertia map, the inertia rebuilt from reflected masses, and the orientation-blind
    sandwich."""
    # Inertia is a duality map taking the body's turning rate to its intrinsic angular momentum.
    inertia = (masses * points).regressive(points.commutator(Rate)).sum(axis=0)  # [] Momentum- <- Rate+
    momentum = inertia(rate)                                                    # [] Momentum-
    energy = (rate & momentum) / 2                                              # [] Scalar+

    # Reflecting the inertia map on both input and output agrees with reflecting the body.
    reflected_rate = mirror >> rate                                             # [] Rate+
    reflected_inertia = mirror >> inertia(mirror << Rate)                       # [] Momentum- <- Rate+
    reflected_energy = (reflected_rate & reflected_inertia(reflected_rate)) / 2  # [] Scalar+

    # Rebuilding the inertia from the reflected masses yields the same map and energy.
    reflected_points = mirror >> points                                         # [...] Point-
    rebuilt_inertia = (masses * reflected_points).regressive(reflected_points.commutator(Rate)).sum(axis=0)  # [] Momentum- <- Rate+
    rebuilt_energy = (reflected_rate & rebuilt_inertia(reflected_rate)) / 2     # [] Scalar+

    # Uniform grade-involuted conjugation flips the intrinsic momentum's sign under reflection.
    sandwich_momentum = mirror * momentum.involute() * mirror.inverse()         # [] Momentum-
    sandwich_rate = mirror * rate.involute() * mirror.inverse()                 # [] Rate+
    sandwich_energy = (sandwich_rate & sandwich_momentum) / 2                   # [] Scalar+
    return stack([energy, reflected_energy, rebuilt_energy, sandwich_energy])   # [4] Scalar+


def motion(points: Point, rate: Rate, mirror: Plane, times: np.ndarray) -> tuple[Point, Point]:
    """Turning `points` (`[n_masses] Point-`) at `rate` across `times` (`[n_frames]`) alongside their reflection in
    `mirror` produces `[n_frames, 2, n_masses] Point-` positions and velocities."""
    rotors = (rate * (-times[:, None] / 2)).exp()                               # [..., 1] Rotor+
    positions = rotors >> points                                                # [...] Point-
    # Each mass on the sphere moves along the commutator of its position with the turning rate.
    velocities = positions.commutator(rate)                                     # [...] Point-
    return stack([positions, mirror >> positions], axis=1), stack([velocities, mirror >> velocities], axis=1)
