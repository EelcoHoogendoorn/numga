"""Two-view epipolar geometry recovers the relative pose of two cameras and the 3D positions of observed points from corresponding sight rays.

Two camera sight rays intersect in space when they are coplanar, which holds when their regressive product `rays_1 & (motor >> rays_2)` vanishes. Perturbing the relative camera motor by a body-frame twist varies the rotated rays by their commutator with the twist, yielding Gauss-Newton normal equations on the six-dimensional twist space. Once the relative pose is known, each sight ray defines a squared point-to-line distance quadric `(ray & Point)`, and summing those quadrics across the two views yields a quadratic form on points whose vertex is the triangulated 3D landmark.

In 3D Projective Geometric Algebra (`PGA3D`, signature `x+y+z+w0`), planes are vectors, lines and twists are bivectors, points are antivectors (trivectors), and rigid motions are even-grade motors.

In the matrix notation of projective vision this reconstruction reads as the eight-point essential matrix constraint, its singular-value decomposition into candidate rotation-translation pairs, and linear triangulation.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import Extensor, NumpyContext
from numga.algebras import PGA3D

# --- algebra and types -----------------------------------------------------
ga = PGA3D
ctx = NumpyContext(ga)
mv = ctx.multivector

Scalar = ga.gatype.scalar()
Point = ga.gatype.antivector()
Line = ga.gatype.antibivector()
Plane = ga.gatype.vector()
Motor = ga.gatype.rotor()
Twist = ga.gatype.bivector()
Camera = ga.gatype((Point, Point))                                       # Point <- Point
# Camera 2 as a rigid body, by which a step is measured: unit masses a unit from its centre along each axis.
BODY = mv.point(np.concatenate([np.eye(3), -np.eye(3)]))                 # [masses] Point


# --- math ------------------------------------------------------------------
def refine(rays_1: Line, rays_2: Line, motor: Motor) -> Motor:
    """One Gauss-Newton step on the relative camera motor, in camera 1's frame, toward crossing sight rays.

    The rays are calibrated sight rays, each in its own camera's frame. The motor needs a nonzero
    baseline; the baseline's length is not observable and stays close to the given one.
    """
    # Camera 2's kinetic energy on twists in its own frame, to measure the step by.
    kinetic = Twist & (BODY & BODY.commutator(Twist)).sum(axis=0)        # [] Scalar <- (Twist, Twist)
    # Two lines meet (are coplanar) iff their regressive product vanishes. For two lines
    # it is a number, one residual per ray pair.
    residuals = rays_1 & (motor >> rays_2)                               # [rays] Scalar

    # A twist moves a line by their commutator, so the residual's rate of change under a
    # twist is a linear form on twists, its Jacobian.
    jacobian = rays_1 & Twist.commutator(motor >> rays_2)                # [rays] Scalar <- Twist

    # Gauss-Newton normal equations as forms on twists, summed over ray pairs. The residual
    # is already a number, so squaring it needs no metric: the curvature is the Jacobian
    # form times itself, and the gradient the residual times the Jacobian form.
    curvature = (jacobian * jacobian).sum(axis=0)                        # [] Scalar <- (Twist, Twist)
    descent = -(residuals * jacobian).sum(axis=0)                        # [] Scalar <- Twist

    # Solve for the 5 observable degrees of freedom, measuring a step by the kinetic energy
    # of camera 2. The flattest mode is the unobservable scale, which moves camera 2 along the
    # baseline; rcond drops it by its curvature per unit energy, and the step has no
    # component along it in that energy, so near the solution the baseline keeps its length.
    energy = kinetic(motor << Twist, motor << Twist)                     # [] Scalar <- (Twist, Twist)
    step = curvature.lstsq(descent, energy, rcond=1e-4)                  # [] Twist
    return ((step * 0.5).exp() * motor).normalized()                     # [] Motor


def triangulate(rays_1: Line, rays_2: Line, motor: Motor) -> Point:
    """The world points the sight rays of both cameras point at, in camera 1's frame, given camera 2's
    motor."""
    # ray & Point is the plane through a ray and an unknown point, and its squared norm is the point's
    # squared distance from the ray. The world point is the vertex of the distance quadrics summed over
    # both views. A dyad on the point's weight pins the scale without moving the vertex, so it is one solve.
    rays = Extensor.stack([rays_1, motor >> rays_2])                     # [views, rays] Line
    distances = (rays & Point).scalar_norm_squared().sum(axis=0)         # [rays] Scalar <- (Point, Point)
    weight = mv.w & Point                                                # [] Scalar <- Point
    return (distances + weight * weight).solve(weight).normalized()      # [rays] Point


def reconstruction(rays_1: Line, rays_2: Line, motor: Motor) -> Iterator[tuple[Motor, Point]]:
    """Jointly solve for the relative camera motor and the world points, from a first guess of the motor on,
    one Gauss-Newton step at a time, without end: each motor and the world points it implies."""
    while True:
        yield motor, triangulate(rays_1, rays_2, motor)
        motor = refine(rays_1, rays_2, motor)
