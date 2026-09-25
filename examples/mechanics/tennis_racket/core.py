"""The tennis racket theorem, or medial axis theorem.

The Dzhanibekov effect / intermediate axis theorem: a rigid body with 3 distinct principal
moments of inertia rotates stably around its major and minor axes, but tumbles unstably
around its intermediate (medial) axis.

The algebra is not fixed here. `ga` is supplied per instance, by
`examples.instantiate("examples.mechanics.tennis_racket.core", Algebra.from_pqr(p, 0, 0))`,
so the same body spins in any dimension p.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import numpy as np

from numga import Algebra, Extensor, NumpyContext
from examples.mechanics import lie_integrators as lie

# Supplied by examples.instantiate.
ga: Algebra
ctx = NumpyContext(ga, dtype=np.float64)
mv = ctx.multivector

Scalar = ga.gatype.scalar()
Motor = ga.gatype.rotor()
Rate = ga.gatype.bivector()
Forque = ga.gatype.antibivector()
Inertia = ga.gatype((Forque, Rate))
InverseInertia = ga.gatype((Rate, Forque))


class Body(NamedTuple):
    """Rigid body dynamical state in Geometric Algebra."""
    motor: Motor
    rate: Rate
    inertia: Inertia
    inertia_inv: InverseInertia


def racket(seed: int) -> Body:
    """Batched bodies spinning in each independent bivector plane.

    The body is a rectangular cuboid with distinct axis lengths 1, 2, 3, ... Works for p of
    2, 3, 4 and 5. For `p > 3` some medial axes become seemingly chaotic, and some medial
    axes stabilize.
    """
    p = ga.dimension
    bits = (np.arange(2**p)[:, None] & (1 << np.arange(p))) > 0
    corners = (2.0 * bits - 1.0) * (np.arange(p) + 1.0)
    points = mv.vector(corners).dual()

    planes = len(Rate.output_subspace)
    inertia, inertia_inv = lie.inertia_from_points(points[None, :].broadcast_to((planes, len(corners))))
    motor = mv.rotor().broadcast_to(planes)

    # Initial spin in each principal plane with a small perturbation
    rng = np.random.default_rng(seed)
    rate = mv.bivector(np.eye(planes) + rng.normal(scale=1e-5, size=(planes, planes)))
    return Body(motor, rate, inertia, inertia_inv)


def simulate(body: Body, step: Callable, dt: float, steps: int) -> tuple[Motor, Rate]:
    """Torque-free motion: motors and rates stacked over time, starting with the initial state."""
    motors, rates = [body.motor], [body.rate]
    for _ in range(steps):
        motor, rate = step(motors[-1], rates[-1], body.inertia, body.inertia_inv, dt, lie.free)
        motors.append(motor)
        rates.append(rate)
    return Extensor.stack(motors), Extensor.stack(rates)


def momentum_drift(motors: Motor, rates: Rate, inertia: Inertia) -> Scalar:
    """Relative drift of the world-frame angular momentum, per time step and body."""
    # World-frame angular momentum.
    momenta = motors >> inertia(rates)
    change = momenta - momenta[0]
    # The squared magnitude of a momentum is the scalar part of `momentum * ~momentum`, in any dimension.
    return ((change * ~change).select[0] / (momenta[0] * ~momenta[0]).select[0]).square_root()
