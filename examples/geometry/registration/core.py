"""Point-cloud registration recovers the rigid motor aligning two corresponding point sets from the one-sided motor eigenproblem.

A motor carrying `source` onto `target` satisfies `target * Motor - Motor * source == 0`, which is linear in the motor, so the whole motion, turn and shift together, is the flattest direction of one quadratic form on motors. Measured against the motor's own metric, which in flat space has no unit for a shift, the eigensolver finds for each turn the shift that fits best: centring, without an origin, so that moving the whole scene moves the fit with it.

The algebra is not fixed here. `ga` is supplied per instance, by `examples.instantiate("examples.geometry.registration.core", PGA3D)` for flat space (`w0x+y+z+`) or with `Algebra("w+x+y+z+")` for the three-sphere, and the same lines serve both.
"""

from __future__ import annotations

import numpy as np

from numga import Algebra
from numga.backend.context import Context


# Supplied by examples.instantiate.
ga: Algebra
ctx: Context
mv = ctx.multivector
Point = ga.gatype.antivector()                      # Antivector-
Bivector = ga.gatype.bivector()                     # Bivector+
Motor = ga.gatype.rotor()                           # Rotor+


# --- math -----------------------------------------------------------------------------
def fit_motor(source: Point, target: Point) -> Motor:
    """The fitted motor is the flattest direction of the summed squared residual of the one-sided equations."""
    source, target = source.normalized(), target.normalized()                  # [points] Point-, Point-
    # target * Motor == Motor * source is linear in the motor: the residual is a map on motors.
    residual = target * Motor - Motor * source                                 # [points] Antivector- <- Motor+

    # Where the metric is degenerate on ideal blades, the bulk norm alone would discard those planes.
    # Adding the weight norm, the bulk norm of the complement, sums the squares of every coefficient.
    bulk = residual.scalar_norm_squared()                                      # [points] Scalar+ <- (Motor+, Motor+)
    weight = residual.dual().scalar_norm_squared()                             # [points] Scalar+ <- (Motor+, Motor+)
    misfit = (bulk + weight).sum(axis=0)                                       # [] Scalar+ <- (Motor+, Motor+)
    # Against the motor's own metric, which in flat space has no unit for a shift, eigh picks for each turn
    # the shift that fits best: centring, without an origin.
    values, motors = misfit.eigh()                                             # [modes] Scalar+, [modes] Motor+
    return motors[values.argmin()].normalized()                                # [] Motor+


# --- plumbing -------------------------------------------------------------------------
def cloud(n_points: int, spread: np.ndarray, rng: np.random.Generator) -> Point:
    """The cloud's points are spread anisotropically, by the given spread along each coordinate."""
    coords = rng.normal(size=(n_points, ga.dimension - 1)) * spread
    return mv.antivector(np.column_stack((coords, np.ones(n_points)))).normalized()


def jitter(points: Point, sigma: float, rng: np.random.Generator) -> Point:
    """Each point is perturbed by an independent Gaussian motor."""
    n_planes = len(Bivector.output_subspace)
    noise = mv.bivector(rng.normal(scale=sigma / 2, size=(*points.shape, n_planes)))
    return noise.exp() >> points
