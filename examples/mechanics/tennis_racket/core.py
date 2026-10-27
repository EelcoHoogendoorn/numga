"""The tennis racket theorem: a body spun about its intermediate axis tumbles.

A rigid body with three distinct principal moments of inertia spins stably about its major and minor
axes, but about its intermediate axis the smallest disturbance grows, and the body flips over and back,
the Dzhanibekov effect. A rate is a bivector, the plane the body turns in, and a body has a principal
plane for every plane of its space; the same racket spins in any dimension, one body per principal plane.

The algebra is not fixed here. `ga` is supplied per instance, by
`examples.instantiate("examples.mechanics.tennis_racket.core", Algebra.from_pqr(p, 0, 0))`,
so the same body spins in any dimension p.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from itertools import islice
from typing import NamedTuple

import numpy as np

from numga import Algebra, stack
from numga.backend.context import Context
from examples.mechanics import lie_integrators as lie

# Supplied by examples.instantiate.
ga: Algebra
ctx: Context
mv = ctx.multivector

Scalar = ga.gatype.scalar()                         # Scalar+
Motor = ga.gatype.rotor()                           # Rotor+
Rate = ga.gatype.bivector()                         # Bivector+
Forque = ga.gatype.antibivector()                   # Antibivector-
Inertia = ga.gatype((Forque, Rate))
InverseInertia = ga.gatype((Rate, Forque))


class Body(NamedTuple):
    """A rigid body: where it is, how fast it turns, and its inertia and inverse inertia."""
    motor: Motor
    rate: Rate
    inertia: Inertia
    inverse_inertia: InverseInertia


def racket(seed: int) -> Body:
    """One body for each principal plane, spun in that plane with a small disturbance.

    The body is the corners of a box with distinct half-lengths 1, 2, 3, and so on. Beyond three
    dimensions some intermediate planes tumble irregularly, and some become stable.
    """
    dimension = ga.dimension
    # Each corner's signs along the axes, from the bits of its index, times the half-lengths.
    signs = 2.0 * ((np.arange(2**dimension)[:, None] & (1 << np.arange(dimension))) > 0) - 1.0
    corners = mv.vector(signs * (np.arange(dimension) + 1.0)).dual()                        # [corners] Antivector-
    n_planes = len(Rate.output_subspace)
    inertia, inverse_inertia = lie.inertia_from_points(corners[None, :].broadcast_to((n_planes, len(signs))))
    motor = mv.rotor().broadcast_to(n_planes)
    rng = np.random.default_rng(seed)
    rate = mv.bivector(np.eye(n_planes) + rng.normal(scale=1e-5, size=(n_planes, n_planes)))   # [bodies] Rate+
    return Body(motor, rate, inertia, inverse_inertia)


def motion(body: Body, step: Callable, dt: float) -> Iterator[tuple[Motor, Rate]]:
    """Torque-free motion from the body's state, one step at a time, without end."""
    motor, rate = body.motor, body.rate
    while True:
        yield motor, rate
        motor, rate = step(motor, rate, body.inertia, body.inverse_inertia, dt, lie.free)


def simulate(body: Body, step: Callable, dt: float, n_steps: int) -> tuple[Motor, Rate]:
    """The motors and rates of the first steps, stacked over time, the initial state first."""
    motors, rates = zip(*islice(motion(body, step, dt), n_steps + 1))
    return stack(motors), stack(rates)                                                      # [steps + 1, bodies] Motor+, Rate+


def momentum_drift(motors: Motor, rates: Rate, inertia: Inertia) -> Scalar:
    """Relative drift of the world-frame angular momentum, per time step and body."""
    # World-frame angular momentum.
    momenta = motors >> inertia(rates)
    change = momenta - momenta[0]
    # The squared magnitude of a momentum is the scalar part of `momentum * ~momentum`, in any dimension.
    return (change.scalar_norm_squared() / momenta[0].scalar_norm_squared()).square_root()
