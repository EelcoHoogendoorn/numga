"""Point-cloud registration recovers the rigid motor aligning two corresponding point sets from the one-sided motor eigenproblem.

A motor carrying each source point onto its target satisfies `target * Motor - Motor * source == 0`, which is linear in the motor, so the whole motion, turn and shift together, is the flattest direction of one quadratic form on motors. Measured against the motor's own metric, which in flat space has no unit for a shift, the eigensolver finds for each turn the shift that fits best, centring the clouds without picking an origin so that moving the whole scene moves the fit with it.

The algebra stays open at module scope and `examples.instantiate` supplies `ga` per scene, running the same lines in three-dimensional projective space and on the three-sphere.
"""

from __future__ import annotations

import numpy as np

from numga import Algebra
from numga.backend.context import Context


# Supplied by examples.instantiate.
ga: Algebra
ctx: Context
mv = ctx.multivector
Point = ga.gatype.antivector()                      # Point-
# Coordinate planes through the origin; the plane w reads a point's weight.
Upright = ga.gatype.vector().derive.within(Algebra("x+y+z+"))  # Plane+
Bivector = ga.gatype.bivector()                     # Bivector+
Motor = ga.gatype.rotor()                           # Motor+


# --- math -----------------------------------------------------------------------------
def fitted_motor(source: Point, target: Point) -> Motor:
    """The one-sided motor eigenproblem recovers the unit motor (`[] Motor+`) aligning `source` onto `target` (`[...] Point-`)."""
    source, target = source.normalized(), target.normalized()                  # [...] Point-, [...] Point-
    # target * Motor == Motor * source is linear in the motor, giving a linear map on motors per pair.
    residual = target * Motor - Motor * source                                 # [...] Point- <- Motor+

    # Where the metric is degenerate on ideal blades, the bulk norm alone discards those planes;
    # adding the bulk norm of the complement sums the squares of every coefficient.
    bulk = residual.scalar_norm_squared()                                      # [...] Scalar+ <- (Motor+, Motor+)
    weight = residual.dual().scalar_norm_squared()                             # [...] Scalar+ <- (Motor+, Motor+)
    misfit = (bulk + weight).sum(axis=0)                                       # [] Scalar+ <- (Motor+, Motor+)
    # Against the motor's own metric, which in flat space has no unit for a shift, eigh picks for each turn
    # the shift that fits best, centring the clouds without choosing an origin.
    values, motors = misfit.eigh()                                             # [8] Scalar+, [8] Motor+
    return motors[values.argmin()].normalized()                                # [] Motor+


# --- plumbing -------------------------------------------------------------------------
def point_cloud(n_points: int, spread: np.ndarray, rng: np.random.Generator) -> Point:
    """Gaussian coordinates scaled by `spread` form an anisotropic cloud of `n_points` unit points (`[n_points] Point-`)."""
    coordinates = rng.normal(size=(n_points, ga.dimension - 1)) * spread       # [n_points, 3]
    return (mv(Upright, coordinates) + mv.w).dual().normalized()               # [n_points] Point-


def noisy_cloud(points: Point, sigma: float, rng: np.random.Generator) -> Point:
    """Independent Gaussian bivector generators of scale `sigma / 2` perturb each point (`[...] Point-`)."""
    n_planes = len(Bivector.output_subspace)
    generators = mv.bivector(rng.normal(scale=sigma / 2, size=(*points.shape, n_planes)))  # [...] Bivector+
    perturbations = generators.exp()                                           # [...] Motor+
    return perturbations >> points                                             # [...] Point-
