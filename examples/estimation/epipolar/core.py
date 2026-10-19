"""Two-view epipolar geometry recovers the relative pose of two cameras and the 3D positions of observed points from corresponding sight rays.

Two camera sight rays intersect in space when they are coplanar, which holds when their regressive product `rays_1 & (motor >> rays_2)` vanishes. Perturbing the relative camera motor by a body-frame twist varies the rotated rays by their commutator with the twist, yielding Gauss-Newton normal equations on the six-dimensional twist space. Once the relative pose is known, each sight ray defines a squared point-to-line distance quadric `(ray & Point)`, and summing those quadrics across the two views yields a quadratic form on points whose smallest eigenvector is the triangulated 3D landmark.

In 3D Projective Geometric Algebra (`PGA3D`, signature `x+y+z+w0`), planes are vectors, lines and twists are bivectors, points are antivectors (trivectors), and rigid motions are even-grade motors.

In the matrix notation of projective vision this reconstruction reads as the eight-point essential matrix constraint, its singular-value decomposition into candidate rotation-translation pairs, and linear triangulation.
"""

from __future__ import annotations

import numpy as np

from numga import NumpyContext
from numga.algebras import PGA3D

# --- algebra and types -----------------------------------------------------
ga = PGA3D
ctx = NumpyContext(ga)
mv = ctx.multivector

Point = ga.gatype.antivector()
Line = ga.gatype.antibivector()
Plane = ga.gatype.vector()
Motor = ga.gatype.rotor()
Twist = ga.gatype.bivector()
RayQuadric = ga.gatype((ga.gatype.scalar(), Point, Point))
Camera = ga.gatype((Point, Point))


# --- plumbing --------------------------------------------------------------
def house_landmarks() -> Point:
    """World landmarks: the corners of a wireframe house and a grid of ground markers."""
    # Base cube corners (8 points):
    cube = np.array([
        [-0.5, -0.4, 3.0],
        [ 0.5, -0.4, 3.0],
        [ 0.5,  0.4, 3.0],
        [-0.5,  0.4, 3.0],
        [-0.5, -0.4, 4.0],
        [ 0.5, -0.4, 4.0],
        [ 0.5,  0.4, 4.0],
        [-0.5,  0.4, 4.0],
    ])
    # Roof ridge and apex points (4 points):
    roof = np.array([
        [ 0.0, -0.4, 4.6],
        [ 0.0,  0.4, 4.6],
        [-0.25, 0.0, 4.3],
        [ 0.25, 0.0, 4.3],
    ])
    # Additional distributed surface / ground markers (12 points):
    grid_x, grid_y = np.meshgrid(np.linspace(-0.8, 0.8, 4), np.linspace(-0.6, 0.6, 3))
    ground = np.stack([grid_x.ravel(), grid_y.ravel(), np.full(12, 2.5)], axis=-1)
    return mv.point(np.concatenate([cube, roof, ground], axis=0))


# --- math ------------------------------------------------------------------
def reconstruct(
    rays_1: Line,
    rays_2: Line,
    motor: Motor,
    iterations: int,
) -> tuple[Motor, Point]:
    """Jointly solve for the relative camera motor and the world points, in camera 1's frame.

    The rays are calibrated sight rays, each in its own camera's frame. The initial motor
    needs a nonzero baseline; the baseline's length is not observable and stays close to the
    given one.
    """
    # Camera 2 as a rigid body, unit masses a unit from its centre, and its kinetic energy on
    # twists in its own frame, to measure the steps by:
    body = mv.point(np.concatenate([np.eye(3), -np.eye(3)]))               # [6] Point
    kinetic = Twist & (body & body.commutator(Twist)).sum(axis=0)          # [] Scalar <- (Bivector, Bivector)
    for _ in range(iterations):
        # Two lines meet (are coplanar) iff their regressive product vanishes. For two lines
        # it is a number, one residual per ray pair:
        res = rays_1 & (motor >> rays_2)                  # [n_rays] Scalar

        # Infinitesimal variation: a twist acts on lines via the commutator.
        # Leaving the Twist slot open yields the Jacobian, a linear form on twists:
        j = rays_1 & Twist.commutator(motor >> rays_2)    # [n_rays] Scalar <- Bivector

        # Gauss-Newton normal equations as forms on twists, summed over ray pairs. The residual
        # is already a number, so squaring it needs no metric: the curvature is the Jacobian
        # form times itself, and the gradient the residual times the Jacobian form:
        h = (j * j).sum(axis=0)                           # [] Scalar <- (Bivector, Bivector)
        rhs = -(res * j).sum(axis=0)                      # [] Scalar <- Bivector

        # Solve for the 5 observable degrees of freedom, measuring a step by the kinetic energy
        # of camera 2. The flattest mode is the unobservable scale, which moves camera 2 along the
        # baseline; rcond drops it by its curvature per unit energy, and the step has no
        # component along it in that energy, so near the solution the baseline keeps its length:
        metric = kinetic(motor << Twist, motor << Twist)  # [] Scalar <- (Bivector, Bivector)
        step = h.lstsq(rhs, metric, rcond=1e-4)           # [] Bivector
        motor = (step * 0.5).exp() * motor                # [] Motor

    # 3D points implied by converged sight rays: ray & Point is the plane through a ray and
    # an unknown point, and its squared norm is the point's squared distance from the ray.
    # The world point is the vertex of the summed ray-distance quadrics. A dyad on the point's
    # weight pins the scale without moving the vertex, so it is one solve:
    aligned_rays_2 = motor >> rays_2
    q1 = (rays_1 & Point) | (rays_1 & Point)
    q2 = (aligned_rays_2 & Point) | (aligned_rays_2 & Point)
    weight = mv.w & Point                               # [] Scalar <- Point
    points = (q1 + q2 + weight * weight).solve(weight)  # [n] Point
    return motor, points.normalized()
