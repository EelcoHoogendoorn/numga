"""Angular momentum, kinetic energy, torque, and mechanical power of a spinning wheel in the plane and in space.

In the Euclidean algebras `VGA2D` (`x+y+`) and `VGA3D` (`x+y+z+`), mass positions relative to the hub and applied pushes
are modelled as intrinsic antivectors, while the wheel's turning rate is modelled as the extrinsic bivector of its
turning plane. Each mass moves with velocity `point.commutator(rate)`. Joining each mass position with its
mass-weighted velocity sums the wheel's inertia map, and joining the handle position with the applied push forms the
intrinsic torque; pairing the momentum and torque against the turning rate gives the kinetic energy
`(momentum & rate) / 2` and mechanical power `torque & rate` in both dimensions and under reflection. The algebra `ga`
is supplied per instance by `examples.instantiate("examples.orientation.spinning_wheel.core", VGA2D)` or `VGA3D`.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import Algebra
from numga.backend.context import Context

# Supplied by examples.instantiate.
ga: Algebra
ctx: Context
mv = ctx.multivector

Point = ga.gatype.antivector()                                                  # Antivector-
Rate = ga.gatype.bivector()                                                     # Bivector+
Momentum = ga.gatype.antibivector()                                             # Antibivector-
Torque = ga.gatype.antibivector()                                               # Antibivector-
Scalar = ga.gatype.scalar()                                                     # Scalar+


# --- math -----------------------------------------------------------------------------
def readings(points: Point, masses: np.ndarray, handles: Point, pushes: Point,
             rates: Rate) -> tuple[Momentum, Scalar, Torque, Scalar]:
    """Mass positions `points` (`[n_wheels, n_masses] Point-`), `handles` (`[n_wheels] Point-`), `pushes` (`[n_wheels] Point-`),
    and `rates` (`[n_wheels] Rate+`) determine each wheel's angular momentum, kinetic energy, torque, and mechanical power."""
    # Joining each mass position with its mass-weighted velocity sums the wheel's rate-to-momentum inertia map.
    inertia = points.regressive(points.commutator(Rate) * masses).sum(axis=-1)  # [n_wheels] Momentum- <- Rate+
    momenta = inertia(rates)                                                    # [n_wheels] Momentum-
    energies = (momenta & rates) / 2                                            # [n_wheels] Scalar+
    # Joining the handle position with the applied push forms the intrinsic torque paired against the rate.
    torques = handles & pushes                                                  # [n_wheels] Torque-
    powers = torques & rates                                                    # [n_wheels] Scalar+
    return momenta, energies, torques, powers


def turning(points: Point, pushes: Point, rates: Rate, times: np.ndarray) -> Iterator[tuple[Point, Point, Point]]:
    """Yields turned mass positions (`[n_wheels, n_masses] Point-`), mass velocities, and handle pushes across `times`."""
    for time in times:
        turns = (rates * (-time / 2)).exp()                                     # [n_wheels] Rotor+
        places = turns[:, None] >> points                                       # [n_wheels, n_masses] Point-
        # Each mass moves along its commutator with the turning rate.
        yield places, places.commutator(rates[:, None]), turns >> pushes
