"""Two-view epipolar geometry recovers the relative pose of two cameras and the 3D positions of observed landmarks from corresponding sight rays.

Each sight ray is modelled as a line, the join of a camera's optical centre with an image point, and two sight rays intersect in space when their regressive product `rays_1 & (motor >> rays_2)` vanishes. Because a twist moves a line by their commutator, the rate of change of each crossing residual under a small camera motion is a linear form on twists, and Gauss–Newton refinement solves against the sum of squared forms measured in the second camera's kinetic energy. Once the relative motor is known, a point's squared distance to a sight ray is the quadric `(sight_rays & Point).scalar_norm_squared()`, and each triangulated landmark sits at the vertex of its two rays' summed distance quadrics.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import NumpyContext, dim, stack
from numga.algebras import PGA3D

# --- algebra and types -----------------------------------------------------
ga = PGA3D
mv = NumpyContext(ga, units=True).multivector

Scalar = ga.gatype.scalar()                                                   # Scalar+
Point = ga.gatype.antivector()                                                # Antivector-
Line = ga.gatype.antibivector()                                               # Antibivector-
Motor = ga.gatype.rotor()                                                     # Rotor+
Twist = ga.gatype.bivector()                                                  # Bivector+
# Unit length scale for camera 2's kinetic-energy metric and homogeneous weight pinning.
camera_scale = mv.scalar(1.0, unit=dim.length)                                # [] Scalar+, length
# Six unit masses at `camera_scale` along the three coordinate axes of camera 2 define the kinetic-energy metric on twists.
body_masses = mv.point([[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]], unit=dim.length)  # [6] Point-, 1 | length
kinetic_energy = Twist.regressive(body_masses.regressive(body_masses.commutator(Twist)).sum(axis=0))  # [] Scalar+ <- (Twist+, Twist+), 1 | length^2
point_weight = mv.w & Point                                                   # [] Scalar+ <- Point-


# --- math ------------------------------------------------------------------
def pose_step(rays_1: Line, rays_2: Line, motor: Motor) -> Motor:
    """One Gauss–Newton step from `motor` (`[] Motor+`) in camera 1's frame moves camera 2 toward intersecting sight rays (`[...] Line-`) while dropping the unobservable baseline-scale mode."""
    # Two lines intersect (are coplanar) when their regressive product vanishes, giving one scalar residual per ray pair.
    residuals = rays_1 & (motor >> rays_2)                                    # [...] Scalar-
    # A twist moves a line by their commutator, so differentiating the crossing residual yields a linear form on twists.
    jacobian = rays_1.regressive(Twist.commutator(motor >> rays_2))           # [...] Scalar- <- Twist+
    # Scalar residuals square directly without a metric, forming the Gauss-Newton curvature and descent forms over twists.
    curvature = (jacobian * jacobian).sum(axis=0)                             # [] Scalar+ <- (Twist+, Twist+)
    descent = -(residuals * jacobian).sum(axis=0)                             # [] Scalar+ <- Twist+
    # Measuring twists in camera 2's kinetic energy isolates the zero-curvature scale mode along the baseline for rcond truncation.
    step_metric = kinetic_energy(motor << Twist, motor << Twist)              # [] Scalar+ <- (Twist+, Twist+)
    step_twist = curvature.lstsq(descent, step_metric, rcond=1e-4)            # [] Twist+
    step_motor = (step_twist * 0.5).exp()                                     # [] Motor+
    return (step_motor * motor).normalized()                                  # [] Motor+


def triangulated_points(rays_1: Line, rays_2: Line, motor: Motor) -> Point:
    """Summing the squared-distance quadrics of `rays_1` and `motor >> rays_2` (`[...] Line-`) locates each triangulated landmark (`[...] Point-`) in camera 1's frame."""
    # Joining a ray with a point spans the plane through both whose squared norm equals the point's squared distance to the ray.
    sight_rays = stack([rays_1, motor >> rays_2])                             # [2, ...] Line-
    distance_quadrics = (sight_rays & Point).scalar_norm_squared().sum(axis=0)  # [...] Scalar+ <- (Point-, Point-)
    # Adding the scaled projective weight dyad pins homogeneous scale without shifting the quadric vertex.
    weight_pin = camera_scale * point_weight                                  # [] Scalar+ <- Point-, length
    return (distance_quadrics + weight_pin * weight_pin).solve(camera_scale * weight_pin).normalized()  # [...] Point-, 1 | length


def reconstruction_states(rays_1: Line, rays_2: Line, motor: Motor) -> Iterator[tuple[Motor, Point]]:
    """Yields camera 2's relative pose (`[] Motor+`) and the triangulated landmarks (`[...] Point-`) across successive Gauss–Newton steps."""
    while True:
        yield motor, triangulated_points(rays_1, rays_2, motor)
        motor = pose_step(rays_1, rays_2, motor)
