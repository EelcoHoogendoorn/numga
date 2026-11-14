"""The tennis racket theorem in any Euclidean dimension.

A rigid body with distinct principal moments of inertia spins stably about its major and minor planes, while a small disturbance near an intermediate plane grows into periodic flips known as the Dzhanibekov effect. Because angular rate is a bivector and angular momentum is an antibivector, the same inertia map and Euler equation spin one copy of a box in each principal plane of `p`-dimensional Euclidean space when instantiated via `examples.instantiate`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from itertools import islice

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
Inertia = ga.gatype((Forque, Rate))                 # Forque- <- Rate+
InverseInertia = ga.gatype((Rate, Forque))          # Rate+ <- Forque-


def racket(seed: int) -> tuple[Motor, Rate, Inertia, InverseInertia]:
    """One box per principal plane (`[n_planes]`) starts spinning in that plane with a small disturbance.

    The box consists of unit-mass points at the `2**p` corners with distinct half-lengths `1, 2, ..., p` along the axes. Beyond three dimensions some intermediate planes tumble irregularly and others remain stable.
    """
    dimension = ga.dimension
    n_corners = 2**dimension
    # Each corner's signs along the axes come from the bits of its index, scaled by the half-lengths.
    signs = 2 * ((np.arange(n_corners)[:, None] & (1 << np.arange(dimension))) > 0) - 1
    corners = mv.vector(signs * (np.arange(dimension) + 1)).dual()                              # [n_corners] Antivector-
    n_planes = len(Rate.output_subspace)
    inertia, inverse_inertia = lie.inertia_from_points(corners[None, :].broadcast_to((n_planes, n_corners)))
    motors = mv.rotor().broadcast_to(n_planes)                                                  # [n_planes] Motor+
    rng = np.random.default_rng(seed)
    rates = mv.bivector(np.eye(n_planes) + rng.normal(scale=1e-5, size=(n_planes, n_planes)))   # [n_planes] Rate+
    return motors, rates, inertia, inverse_inertia


def motion(
    motors: Motor,
    rates: Rate,
    inertia: Inertia,
    inverse_inertia: InverseInertia,
    step: Callable,
    dt: float,
) -> Iterator[tuple[Motor, Rate]]:
    """Yields the torque-free motors and rates (`[...] Motor+, Rate+`) one time step after another."""
    while True:
        yield motors, rates
        motors, rates = step(motors, rates, inertia, inverse_inertia, dt, lie.free)


def simulated_trajectory(
    motors: Motor,
    rates: Rate,
    inertia: Inertia,
    inverse_inertia: InverseInertia,
    step: Callable,
    dt: float,
    n_steps: int,
) -> tuple[Motor, Rate]:
    """Torque-free motion over `n_steps` steps stacks into the full trajectory (`[n_steps + 1, ...] Motor+, Rate+`)."""
    step_motors, step_rates = zip(*islice(motion(motors, rates, inertia, inverse_inertia, step, dt), n_steps + 1))
    return stack(step_motors), stack(step_rates)                                                # [n_steps + 1, ...] Motor+, Rate+


def momentum_drift(motors: Motor, rates: Rate, inertia: Inertia) -> Scalar:
    """World-frame angular momentum drifts relative to `motors[0]`, `rates[0]` by this dimensionless fraction (`[...] Scalar+`)."""
    # Carry body-frame angular momentum into the world frame with the motor sandwich.
    momenta = motors >> inertia(rates)                                                          # [...] Forque-
    change = momenta - momenta[0]                                                               # [...] Forque-
    # `scalar_norm_squared` gives the squared norm of an antibivector in any Euclidean dimension.
    return (change.scalar_norm_squared() / momenta[0].scalar_norm_squared()).square_root()      # [...] Scalar+

