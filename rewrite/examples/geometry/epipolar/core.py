"""Two-view epipolar geometry, relative pose, and 3D reconstruction in PGA3D.

In projective geometric algebra (PGA3D), camera sight rays are lines (bivectors).
Two sight rays intersect in 3D if and only if their wedge product vanishes:
    rays_1 ^ rays_2 == 0

For a candidate relative camera motor, both cameras shoot sight rays into the world.
Infinitesimal motor updates follow the Lie-algebra commutator extensor, minimizing
the mutual line intersection error across corresponding screen points without any
separation of rotation and translation.

With the relative pose solved, the 3D world coordinates are implied: each sight ray
measures perpendicular distance to an unknown point via the meet `ray & Point`.
Its transpose square forms a rank-2 distance quadric; summing these quadrics across
cameras produces an extensor whose nullmode is the reconstructed 3D world point.

This module contains the mathematics alone: GATypes and the geometric narrative
in one coherent scope.
"""

from __future__ import annotations

from numga import NumpyContext
from numga.algebras import PGA3D

# --- algebra and types -----------------------------------------------------
ga = PGA3D
ctx = NumpyContext(ga)
mv = ctx.multivector

Point = ga.gatype.antivector()
Line = ga.gatype.bivector()
Motor = ga.gatype.rotor()
Twist = ga.gatype.bivector()
RayQuadric = ga.gatype((Point, Point))


# --- math ------------------------------------------------------------------
def reconstruct(
    rays_1: Line,
    rays_2: Line,
    motor: Motor,
    iterations: int = 10,
) -> tuple[Motor, Point]:
    """Jointly solve for relative camera motor and 3D world coordinates.

    Parameters
    ----------
    rays_1 : Line
        Calibrated sight rays in Camera 1's frame (reference frame).
    rays_2 : Line
        Calibrated sight rays in Camera 2's local frame.
    motor : Motor
        Initial relative camera motor estimate (non-unit baseline required).
    iterations : int
        Number of Lie-algebra Gauss-Newton updates.

    Returns
    -------
    motor : Motor
        Relative camera motor aligning sight rays in Camera 1's frame.
    points : Point
        Reconstructed 3D coordinates in Camera 1's frame.
    """
    for _ in range(iterations):
        # Two lines meet (are coplanar) iff their wedge product vanishes.
        # The residual is a pseudoscalar measuring signed ray separation distance:
        res = rays_1 ^ (motor >> rays_2)                  # [n_rays] Pseudoscalar

        # Infinitesimal variation: an se(3) twist acts on lines via commutator.
        # Leaving the Twist slot open yields the Jacobian map:
        j = rays_1 ^ Twist.commutator(motor >> rays_2)    # [n_rays] Pseudoscalar <- Bivector

        # Accumulate the Gauss-Newton normal equations across all ray pairs:
        h = (j.transpose()(j)).sum(axis=0)                # [] Bivector <- Bivector
        rhs = -(j.transpose()(res)).sum(axis=0)           # [] Bivector <- Pseudoscalar

        # Solve for the 5 observable degrees of freedom; rcond discards the
        # unobservable translation scale gauge mode without Cartesian decomposition:
        step = h.lstsq(rhs, rcond=1e-4)                   # [] Bivector
        motor = (step * 0.5).exp() * motor                # [] Motor

    # 3D points implied by converged sight rays:
    # ray & Point measures distance from each ray to an unknown 3D point.
    # Summing the two ray-distance quadrics, the nullmode yields the world point:
    aligned_rays_2 = motor >> rays_2
    q1 = (rays_1 & Point).transpose()(rays_1 & Point)
    q2 = (aligned_rays_2 & Point).transpose()(aligned_rays_2 & Point)
    values, points = (q1 + q2).eigh()
    idx, = set(values.argmin(axis=-1))
    return motor, points[..., idx].normalized()
