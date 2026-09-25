"""Two-view epipolar geometry, relative pose, and 3D reconstruction in PGA3D.

In projective geometric algebra (PGA3D), camera sight rays are lines (bivectors).
Two sight rays intersect in 3D if and only if their wedge product vanishes:
    rays_1 ^ rays_2 == 0

For a candidate relative camera motor, both cameras shoot sight rays into the world.
Infinitesimal motor updates follow the Lie-algebra commutator extensor, minimizing
the mutual line intersection error across corresponding screen points without any
separation of rotation and translation.

With the relative pose solved, the 3D world coordinates are implied: each sight ray
measures distance to an unknown point via the meet `ray & Point`, a plane whose
squared norm is a rank-2 distance quadric; summing these quadrics across cameras
produces a form whose nullmode is the reconstructed 3D world point.

`textbook.py` holds the matrix baseline this is compared against: the 8-point essential
matrix, its SVD decomposition into four candidate poses, and triangulation by normal
equations.
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
Line = ga.gatype.bivector()
Plane = ga.gatype.vector()
Motor = ga.gatype.rotor()
Twist = ga.gatype.bivector()
RayQuadric = ga.gatype((ga.gatype.scalar(), Point, Point))
Camera = ga.gatype((Point, Point))


def point(coords: np.ndarray) -> Point:
    """Finite points at (..., 3) coordinates: the dual of the homogeneous vector."""
    return (mv("x y z", coords) + mv.w).dual()


def direction(coords: np.ndarray) -> Point:
    """Ideal points: the directions (..., 3), the dual of a weightless vector."""
    return mv("x y z", coords).dual()


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
    return point(np.concatenate([cube, roof, ground], axis=0))


# --- math ------------------------------------------------------------------
def reconstruct(
    rays_1: Line,
    rays_2: Line,
    motor: Motor,
    iterations: int,
) -> tuple[Motor, Point]:
    """Jointly solve for the relative camera motor and the world points, in camera 1's frame.

    The rays are calibrated sight rays, each in its own camera's frame. The initial motor
    needs a nonzero baseline; the baseline's length is not observable and stays as given.
    """
    for _ in range(iterations):
        # Two lines meet (are coplanar) iff their wedge product vanishes. The residual is
        # a pseudoscalar, one number per ray pair, read as a scalar through the complement:
        res = (rays_1 ^ (motor >> rays_2)).dual()         # [n_rays] Scalar

        # Infinitesimal variation: an se(3) twist acts on lines via the commutator.
        # Leaving the Twist slot open yields the Jacobian, a linear form on twists:
        j = (rays_1 ^ Twist.commutator(motor >> rays_2)).dual()   # [n_rays] Scalar <- Bivector

        # Gauss-Newton normal equations as forms on twists, summed over ray pairs. The residual
        # is already a number, so squaring it needs no metric: the curvature is the Jacobian
        # form times itself, and the gradient the residual times the Jacobian form:
        h = (j * j).sum(axis=0)                           # [] Scalar <- (Bivector, Bivector)
        rhs = -(res * j).sum(axis=0)                      # [] Scalar <- Bivector

        # Solve for the 5 observable degrees of freedom; rcond discards the
        # unobservable translation scale gauge mode without Cartesian decomposition:
        step = h.lstsq(rhs, rcond=1e-4)                   # [] Bivector
        motor = (step * 0.5).exp() * motor                # [] Motor

    # 3D points implied by converged sight rays: ray & Point is the plane through a ray and
    # an unknown point, and its squared norm is the point's squared distance from the ray.
    # Summing the two ray-distance quadrics, the nullmode yields the world point:
    aligned_rays_2 = motor >> rays_2
    q1 = (rays_1 & Point) | (rays_1 & Point)
    q2 = (aligned_rays_2 & Point) | (aligned_rays_2 & Point)
    # Against the point's own metric only the weight is measured, so the least mode minimizes the
    # summed squared distance of a unit-weight point. That metric is singular on the bulk; the
    # general eigenproblem sends those modes to infinity and the least finite mode is real.
    values, points = (q1 + q2).eig()                    # [n, n_modes] Scalar, [n, n_modes] Point
    least = values.real().argmin(axis=-1)                 # [n] mode index per point
    return motor, points[np.arange(least.shape[0]), least].real().normalized()
