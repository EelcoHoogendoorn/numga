r"""N-camera multi-view reconstruction and bundle adjustment with perspective cone quadrics.

Each camera is represented as a projective transformation extensor `Camera: Point -> Point`.

A pixel measurement on the camera sensor plane has an uncertainty disk quadric.
Pulling this sensor quadric back through the camera projection map forms a true
3D perspective cone quadric ($Plane \leftarrow Point$) whose cross-section naturally widens with depth:
    pullback = camera.transpose()(Plane.dual()).dual_inverse()
    cone = pullback(sensor_quadric(camera))

Multi-view bundle adjustment operates purely on perspective cone quadrics:
1. Triangulating world landmarks by summing perspective cone quadrics across observing cameras.
   The center of the fused quadric (the pole of the plane at infinity w) yields the 3D landmark:
       points = q_fused.inverse()(w).normalized()
   and the fused quadric defines their 3D Gaussian splat precision ellipsoids.
2. Updating camera poses by evaluating the polar plane residuals and Lie-algebra Jacobians
   directly on the perspective cone quadrics.

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
Plane = ga.gatype.vector()
Motor = ga.gatype.rotor()
Twist = ga.gatype.bivector()
Camera = ga.gatype((Point, Point))
Quadric = ga.gatype((Plane, Point))


# --- math ------------------------------------------------------------------
def make_cones(
    cameras: Camera,
    sensor_quadrics: Quadric,
) -> Quadric:
    """Pull sensor measurement quadrics back through camera maps into 3D perspective cones.

    Parameters
    ----------
    cameras : [n_cams] Camera
        Projective camera transformation extensors in local frame.
    sensor_quadrics : [n_points, n_cams] Quadric
        Measurement uncertainty quadrics on each camera's sensor plane.

    Returns
    -------
    cones : [n_points, n_cams] Quadric
        3D perspective cone quadrics in each camera's local frame.
    """
    # Pull back sensor plane quadrics through projective camera map into 3D cones:
    pullback = cameras.transpose()(Plane.dual()).dual_inverse()  # [n_cams] Plane <- Plane
    return pullback(sensor_quadrics(cameras))                    # [n_points, n_cams] Plane <- Point


def triangulate_cones(
    motors: Motor,
    cones: Quadric,
) -> tuple[Point, Quadric]:
    """Triangulate world landmarks by summing perspective cone quadrics across cameras.

    The center of a quadric is the pole of the plane at infinity (w).
    Evaluating the inverted dual quadric on w directly extracts the 3D landmark.

    Parameters
    ----------
    motors : [n_cams] Motor
        Camera poses in world frame.
    cones : [n_points, n_cams] Quadric
        Perspective cone quadrics in each camera local frame.

    Returns
    -------
    points : [n_points] Point
        Reconstructed world landmarks.
    fused_quadrics : [n_points] Quadric
        Fused perspective cone quadrics (3D Gaussian splat precision ellipsoids).
    """
    # Transform local cones to world frame and sum into fused precision quadrics:
    world_cones = motors >> cones(motors << Point)           # [n_points, n_cams] Plane <- Point
    q_fused = world_cones.sum(axis=-1)                       # [n_points] Plane <- Point

    # Invert to dual quadric and evaluate on the plane at infinity (w) to find center:
    points = q_fused.inverse()(mv.w).normalized()            # [n_points] Point
    return points, q_fused


def bundle_adjust(
    initial_motors: Motor,
    local_cones: Quadric,
    iterations: int,
) -> tuple[Motor, Point, Quadric]:
    """Jointly optimize camera poses and landmarks purely via perspective cone quadrics.

    Parameters
    ----------
    initial_motors : [n_cams] Motor
        Initial camera pose estimates.
    local_cones : [n_points, n_cams] Quadric
        Perspective cone quadrics in each camera local frame.
    iterations : int
        Number of alternating Gauss-Newton iterations.

    Returns
    -------
    motors : [n_cams] Motor
        Optimized camera poses.
    points : [n_points] Point
        Reconstructed world landmarks.
    quadrics : [n_points] Quadric
        Fused perspective cone quadrics.
    """
    motors = initial_motors
    for _ in range(iterations):
        # Triangulate world landmarks as poles of infinity from fused quadrics:
        points, q_fused = triangulate_cones(motors, local_cones)

        # Pull world landmarks into local frames and evaluate polar plane residuals:
        local_points = motors << points[:, None]             # [n_points, n_cams] Point
        res = local_cones(local_points)                      # [n_points, n_cams] Plane

        # Pose variation under se(3) twist commutator yields polar plane Jacobians:
        j = -local_cones(Twist.commutator(local_points))     # [n_points, n_cams] Plane <- Twist

        # Accumulate Gauss-Newton normal equations across all observed landmarks:
        h = (j.transpose()(j)).sum(axis=0)                   # [n_cams] Twist <- Twist
        rhs = -(j.transpose()(res)).sum(axis=0)              # [n_cams] Twist

        # Solve Lie-algebra twist steps and anchor camera 0 to fix gauge freedom:
        step = h.lstsq(rhs, rcond=1e-4)                      # [n_cams] Twist
        step = step - step[0]                                # [n_cams] Twist
        motors = motors * (step * 0.5).exp()                 # [n_cams] Motor

    points, q_fused = triangulate_cones(motors, local_cones)
    return motors, points, q_fused
