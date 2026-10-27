"""Two-view epipolar geometry recovers the relative pose of two cameras and the 3D positions of observed points from corresponding sight rays.

Each sight ray is a line, the join of a camera's eye with an image point, and two lines cross in space when their regressive product `rays_1 & (motor >> rays_2)` vanishes. A twist moves a line by their commutator, so the residuals' rates under a small motion are linear forms on twists, and Gauss-Newton steps solve against the sum of their squares, a quadratic form on twists measured by the second camera's kinetic energy. With the pose found, a ray's squared distance to a point is the quadric `(rays & Point).scalar_norm_squared()`, and each landmark is the vertex of its two rays' quadrics summed.

In the matrix notation of computer vision the crossing condition reads as the essential matrix constraint $\\mathbf r_2^{\\mathsf T} E\\, \\mathbf r_1 = 0$ with $E = [\\mathbf t]_\\times R$, solved by the eight-point algorithm and a singular value decomposition into four candidate poses, followed by linear triangulation.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import NumpyContext, stack
from numga.algebras import PGA3D

# --- algebra and types -----------------------------------------------------
ga = PGA3D
mv = NumpyContext(ga).multivector

Scalar = ga.gatype.scalar()                         # Scalar+
Point = ga.gatype.antivector()                      # Antivector-
Line = ga.gatype.antibivector()                     # Antibivector-
Motor = ga.gatype.rotor()                           # Rotor+
Twist = ga.gatype.bivector()                        # Bivector+
# Camera 2 as a rigid body, by which a step is measured: unit masses a unit from its centre along each axis.
BODY = mv.point(np.concatenate([np.eye(3), -np.eye(3)]))                 # [masses] Point-


# --- math ------------------------------------------------------------------
def refine(rays_1: Line, rays_2: Line, motor: Motor) -> Motor:
    """One Gauss-Newton step on the relative camera motor, in camera 1's frame, moves it toward crossing sight rays.

    The rays are calibrated sight rays, each in its own camera's frame. The motor needs a nonzero
    baseline; the baseline's length is not observable and stays close to the given one.
    """
    # Camera 2's kinetic energy on twists in its own frame, to measure the step by.
    kinetic = Twist & (BODY & BODY.commutator(Twist)).sum(axis=0)        # [] Scalar+ <- (Twist+, Twist+)
    # Two lines meet (are coplanar) iff their regressive product vanishes. For two lines
    # it is a number, one residual per ray pair.
    residuals = rays_1 & (motor >> rays_2)                               # [rays] Scalar-

    # A twist moves a line by their commutator, so the residual's rate of change under a
    # twist is a linear form on twists, its Jacobian.
    jacobian = rays_1 & Twist.commutator(motor >> rays_2)                # [rays] Scalar- <- Twist+

    # Gauss-Newton normal equations as forms on twists, summed over ray pairs. The residual
    # is already a number, so squaring it needs no metric: the curvature is the Jacobian
    # form times itself, and the gradient the residual times the Jacobian form.
    curvature = (jacobian * jacobian).sum(axis=0)                        # [] Scalar+ <- (Twist+, Twist+)
    descent = -(residuals * jacobian).sum(axis=0)                        # [] Scalar+ <- Twist+

    # Solve for the 5 observable degrees of freedom, measuring a step by the kinetic energy
    # of camera 2. The flattest mode is the unobservable scale, which moves camera 2 along the
    # baseline; rcond drops it by its curvature per unit energy, and the step has no
    # component along it in that energy, so near the solution the baseline keeps its length.
    energy = kinetic(motor << Twist, motor << Twist)                     # [] Scalar+ <- (Twist+, Twist+)
    step = curvature.lstsq(descent, energy, rcond=1e-4)                  # [] Twist+
    return ((step * 0.5).exp() * motor).normalized()                     # [] Motor+


def triangulate(rays_1: Line, rays_2: Line, motor: Motor) -> Point:
    """Each triangulated world point is the point nearest its sight rays from both cameras, in camera 1's frame."""
    # ray & Point is the plane through a ray and an unknown point, and its squared norm is the point's
    # squared distance from the ray. The world point is the vertex of the distance quadrics summed over
    # both views. A dyad on the point's weight pins the scale without moving the vertex, so it is one solve.
    rays = stack([rays_1, motor >> rays_2])                              # [views, rays] Line-
    distances = (rays & Point).scalar_norm_squared().sum(axis=0)         # [rays] Scalar+ <- (Point-, Point-)
    weight = mv.w & Point                                                # [] Scalar+ <- Point-
    return (distances + weight * weight).solve(weight).normalized()      # [rays] Point-


def reconstruction(rays_1: Line, rays_2: Line, motor: Motor) -> Iterator[tuple[Motor, Point]]:
    """Successive Gauss-Newton steps from an initial motor yield each relative camera motor and the world points it triangulates."""
    while True:
        yield motor, triangulate(rays_1, rays_2, motor)
        motor = refine(rays_1, rays_2, motor)
