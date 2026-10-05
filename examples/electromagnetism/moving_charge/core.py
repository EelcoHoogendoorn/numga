"""The field of a soft charge moving at any speed, from the derivatives of its potential.

The charge is a Plummer-softened point charge: beyond its core it is a point charge, and its density
is smooth and finite at its centre. At rest its potential points along its time direction and falls
off as one over the distance, and the gradients of the potential and of the field at an event are
maps on displacements, written in closed form. A moving charge is the charge at rest seen through
a boost: its gradients are the rest-frame ones with the boost sandwiched around them, the
displacement pulled into the rest frame and the result pushed back out.

Contracting a gradient against an open vector gives the derivative: of the potential, the field
bivector plus the Lorenz-gauge scalar, which vanishes; of the field, the current plus a trivector,
which vanishes. Both halves of Maxwell's equations also follow without a metric, from the incidence
trace of the field's gradient against an open antivector: on the field itself the trace is the
current, on its dual it vanishes. The metric enters only through that dual.

A point charge circling at constant speed radiates. Its potential at an event is set by where the
charge was when it sent the light that reaches the event, the retarded time, and the gradient of
that potential is again a closed-form map: contracting it gives the full field, the bound field
that moves with the charge and the waves it sheds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from numga import NumpyContext
from numga.algebras import STA as ga

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()
Vector = ga.gatype.vector()
Bivector = ga.gatype.bivector()
Antivector = ga.gatype.antivector()
Rotor = ga.gatype.rotor()
Even = ga.gatype.even()
Odd = ga.gatype.odd()
PotentialGradient = ga.gatype((Vector, Vector))              # Vector <- Vector
# The retarded time: steps that always close in, by at least the orbit's speed each, then Newton's.
SETTLE_STEPS = 25
NEWTON_STEPS = 6
FieldGradient = ga.gatype((Bivector, Vector))                # Bivector <- Vector


@dataclass(frozen=True)
class Orbit:
    charge: float
    radius: float
    speed: float                                             # fraction of the speed of light


@dataclass(frozen=True)
class Charge:
    charge: float
    core_radius: float
    boost: Rotor                                             # [...] Rotor, the rest frame seen from the lab


# --- math -----------------------------------------------------------------------------
def potential_gradient(charge: Charge, events: Vector) -> PotentialGradient:
    """The change in the potential for a small displacement from each event."""
    separations, weight, _ = _profile(charge, events)                        # [...] Vector, Scalar
    # At rest the potential points along the rest time and changes with the spatial distance.
    at_rest = weight * mv.t * (separations | Vector)                          # [...] Vector <- Vector
    return charge.boost >> at_rest(charge.boost << Vector)                    # [...] Vector <- Vector


def field_gradient(charge: Charge, events: Vector) -> FieldGradient:
    """The change in the field for a small displacement from each event."""
    separations, weight, weight_change = _profile(charge, events)            # [...] Vector, Scalar, Scalar
    # The radial field changes with the distance and turns with the displacement.
    radial = weight_change * (separations ^ mv.t) * (separations | Vector)    # [...] Bivector <- Vector
    turning = weight * (Vector ^ mv.t)                                        # [...] Bivector <- Vector
    return charge.boost >> (radial + turning)(charge.boost << Vector)         # [...] Bivector <- Vector


def potential_derivative(gradients: PotentialGradient) -> Even:
    """The field bivector plus the Lorenz-gauge scalar."""
    return (Vector * gradients(Vector)).contract()


def field(gradients: PotentialGradient) -> Bivector:
    """The field: the bivector part of the potential's derivative."""
    return (Vector ^ gradients(Vector)).contract()


def field_derivative(gradients: FieldGradient) -> Odd:
    """The current plus a trivector that vanishes: Maxwell's equations through the metric."""
    return (Vector * gradients(Vector)).contract()


def sources(gradients: FieldGradient) -> tuple[Vector, Vector]:
    """Maxwell's equations without a metric: the incidence trace of the field's gradient, which is
    the current, and of its dual's, which vanishes."""
    current = (gradients(Vector) & Antivector).trace(1, 2)                   # [...] Vector
    magnetic_source = (gradients(Vector).dual() & Antivector).trace(1, 2)    # [...] Vector
    return current, magnetic_source


def orbit_potential_gradient(orbit: Orbit, events: Vector) -> PotentialGradient:
    """The change in the circling charge's potential for a small displacement from each event."""
    emitted = retarded(orbit, events)                                        # [...] Scalar
    position, tangent, bend = worldline(orbit, emitted)                      # [...] Vector each
    # The charge's velocity and acceleration in its own time, and the light ray from it to the event.
    dilation = 1 / np.sqrt(1 - orbit.speed**2)
    velocity, acceleration = dilation * tangent, dilation**2 * bend          # [...] Vector each
    ray = events - position                                                  # [...] Vector
    reach = ray | velocity                                                   # [...] Scalar
    # A displacement moves the emission along the ray, which changes the velocity and the reach.
    emission = (ray | Vector) / reach                                        # [...] Scalar <- Vector
    reach_change = (velocity | Vector) + ((ray | acceleration) - 1) * emission   # [...] Scalar <- Vector
    weight = orbit.charge / (4 * np.pi) / reach                              # [...] Scalar
    return weight * (acceleration * emission - velocity * reach_change / reach)   # [...] Vector <- Vector


def worldline(orbit: Orbit, times: Scalar) -> tuple[Vector, Vector, Vector]:
    """The circling charge at each lab time, its change per unit lab time, and that change's change."""
    turning = orbit.speed / orbit.radius                                     # angular rate
    offset = (mv.xy * (-0.5 * turning) * times).exp() >> (mv.x * orbit.radius)   # [...] Vector
    tangent = mv.t + turning * (offset | mv.xy)                              # [...] Vector
    bend = turning * ((tangent - mv.t) | mv.xy)                              # [...] Vector
    return times * mv.t + offset, tangent, bend


def retarded(orbit: Orbit, events: Vector) -> Scalar:
    """The lab time at which the charge sent the light that reaches each event: a few steps that
    move the time back by the distance light covers, which always close in, then Newton's steps."""
    def light(times: Scalar) -> tuple[Scalar, Scalar, Scalar]:
        """How far light from the charge at each time falls short of the event, its rate, and the distance."""
        position, tangent, _ = worldline(orbit, times)                       # [...] Vector each
        across = (events - position) - ((events - position) | mv.t) * mv.t  # [...] Vector
        distance = (-(across | across)).square_root()                        # [...] Scalar
        return (events | mv.t) - times - distance, 1 + (across | tangent) / distance, distance

    times = events | mv.t                                                    # [...] Scalar
    for _ in range(SETTLE_STEPS):
        _, _, distance = light(times)
        times = (events | mv.t) - distance
    for _ in range(NEWTON_STEPS):
        shortfall, rate, _ = light(times)
        times = times + shortfall / rate
    return times


def _profile(charge: Charge, events: Vector) -> tuple[Vector, Scalar, Scalar]:
    """The spatial separation from the charge in its rest frame, the field per unit separation, and
    that weight's change per unit of `separation | displacement`."""
    at_rest = charge.boost << events                                         # [...] Vector
    separations = at_rest - (at_rest | mv.t) * mv.t                          # [...] Vector
    softened = charge.core_radius**2 - (separations | separations)           # [...] Scalar
    weight = charge.charge / (4 * np.pi) / (softened * softened.square_root())   # [...] Scalar
    return separations, weight, 3 * weight / softened


# --- plumbing -------------------------------------------------------------------------
def grid(half_width: float, half_height: float, columns: int, rows: int) -> Vector:
    """Events at lab time zero on a grid in the plane z = 0, at the centres of its cells."""
    x = (np.arange(columns) + 0.5) / columns * 2 * half_width - half_width
    y = (np.arange(rows) + 0.5) / rows * 2 * half_height - half_height
    return (mv.x * x[None, :] + mv.y * y[:, None]).cast(Vector)              # [rows, columns] Vector
