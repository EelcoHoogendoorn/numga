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

import numpy as np

from examples.geometry.multiview.types import (
    Camera,
    Hyperplane,
    Motor,
    Point,
    Quadric,
    Scalar,
    Twist,
    TwistMap,
    w,
)


def sensor_disk(pixels: Point, p0: Point, q_sensor: Quadric) -> Quadric:
    """Translate transverse sensor uncertainty to pixel locations."""
    trans = (pixels / p0).square_root()
    return trans >> q_sensor(trans << Point)


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
    pullback = cameras.transpose()(Hyperplane.dual()).dual_inverse()  # [n_cams] Plane <- Plane
    return pullback(sensor_quadrics(cameras))                         # [n_points, n_cams] Plane <- Point


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

    # Evaluate the inverted dual quadric on the plane at infinity (w) to extract the landmark.
    # Adding the gauge dyad w * (w & Point) regularizes the null metric direction
    # without altering the spatial gradient:
    points = (q_fused + w * (w & Point)).solve(w).normalized()
    return points, q_fused


def depths(cameras: Camera, motors: Motor, points: Point) -> Scalar:
    """Perpendicular distance (depth z) of landmarks from each camera principal plane.

    The principal plane is the pullback of the sensor plane's ideal boundary (w)
    through the camera projection map:
        principal = cameras.transpose()(w.dual()).dual_inverse()
    Its evaluation on a local landmark yields the true perpendicular depth z
    purely within typed PGA without requiring an external optical axis reference.

    Parameters
    ----------
    cameras : [n_cams] Camera
        Projective camera transformation extensors in local frame.
    motors : [n_cams] Motor
        Camera poses in world frame.
    points : [n_points] Point
        Landmarks in world frame.

    Returns
    -------
    z : [n_points, n_cams] Scalar
        Positive perpendicular depth along the optical axis (clamped >= 0.1).
    """
    principal = cameras.transpose()(w.dual()).dual_inverse()
    local_points = motors << points[:, None]
    return (principal & local_points).abs().clip(0.1, None)


def reweight_cones(
    cameras: Camera,
    motors: Motor,
    cones: Quadric,
    iterations: int = 3,
) -> Quadric:
    """Reweight perspective cone quadrics into pixel units via Sampson depth scaling.

    Starts from algebraic cone quadrics and iteratively scales each cone by
    1 / z^2 using landmark depths, converging to true inverse pixel variance units.

    Parameters
    ----------
    cameras : [n_cams] Camera
        Projective camera transformation extensors in local frame.
    motors : [n_cams] Motor
        Camera poses in world frame.
    cones : [n_points, n_cams] Quadric
        Perspective cone quadrics in each camera local frame.
    iterations : int
        Number of IRLS depth reweighting iterations. Defaults to 3.

    Returns
    -------
    scaled_cones : [n_points, n_cams] Quadric
        Perspective cone quadrics scaled to true inverse pixel variance units.
    """
    weighted_cones = cones
    for _ in range(iterations):
        points, _ = triangulate_cones(motors, weighted_cones)
        z = depths(cameras, motors, points)
        weighted_cones = cones / (z ** 2)
    return weighted_cones


def triangulate_reweighted(
    cameras: Camera,
    motors: Motor,
    cones: Quadric,
    iterations: int = 3,
) -> tuple[Point, Quadric]:
    """Iteratively reweighted least squares (IRLS) triangulation in pixel units.

    Starts from the closed-form algebraic solve (z = 1) and reweights each cone
    by 1 / z^2 using depths from the previous iterate, converging to the Sampson
    geometric minimum.

    Parameters
    ----------
    cameras : [n_cams] Camera
        Projective camera transformation extensors in local frame.
    motors : [n_cams] Motor
        Camera poses in world frame.
    cones : [n_points, n_cams] Quadric
        Perspective cone quadrics in each camera local frame.
    iterations : int
        Number of IRLS reweighting iterations. Defaults to 3.

    Returns
    -------
    points : [n_points] Point
        Reconstructed world landmarks at the Sampson geometric minimum.
    fused_quadrics : [n_points] Quadric
        Fused perspective cone quadrics in true inverse pixel variance units.
    """
    return triangulate_cones(motors, reweight_cones(cameras, motors, cones, iterations))


def bundle_adjust(
    initial_motors: Motor,
    local_cones: Quadric,
    iterations: int,
    damping: float = 0.9,
    anchors: tuple[int, ...] = (0,),
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
    damping : float
        Step damping factor. Defaults to 0.9.
    anchors : tuple[int, ...]
        Indices of cameras to anchor as fixed gauge reference frames. Defaults to (0,).

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
        points, _ = triangulate_cones(motors, local_cones)

        # Pull world landmarks into local frames and evaluate polar plane residuals:
        local_points = motors << points[:, None]             # [n_points, n_cams] Point
        res = local_cones(local_points)                      # [n_points, n_cams] Plane

        # Pose variation under se(3) twist commutator yields polar plane Jacobians:
        j = -local_cones(Twist.commutator(local_points))     # [n_points, n_cams] Plane <- Twist

        # Accumulate Gauss-Newton normal equations across all observed landmarks:
        h = (j.transpose()(j)).sum(axis=0)                   # [n_cams] Twist <- Twist
        rhs = -(j.transpose()(res)).sum(axis=0)              # [n_cams] Twist

        # Solve Lie-algebra twist steps and anchor reference cameras to fix gauge freedom:
        step = h.lstsq(rhs, rcond=1e-4)                      # [n_cams] Twist
        for a in anchors:
            step = step.at[a].set(step[a] * 0)
        motors = motors * (step * (0.5 * damping)).exp()     # [n_cams] Motor

    points, q_fused = triangulate_cones(motors, local_cones)
    return motors, points, q_fused


def schur_bundle_adjust(
    cameras: Camera,
    initial_motors: Motor,
    local_cones: Quadric,
    iterations: int = 10,
    damping: float = 0.7,
    anchors: tuple[int, ...] = (0,),
) -> tuple[Motor, Point, Quadric, TwistMap]:
    """Schur-complement bundle adjustment folding landmark compliance into camera poses.

    Subtracts landmark compliance (the inverse fused quadric) from the camera stiffness,
    eliminating landmark degrees of freedom to first order and yielding the camera pose covariance.

    Parameters
    ----------
    cameras : [n_cams] Camera
        Projective camera transformation extensors in local frame.
    initial_motors : [n_cams] Motor
        Initial camera pose estimates.
    local_cones : [n_points, n_cams] Quadric
        Perspective cone quadrics in each camera local frame.
    iterations : int
        Maximum number of Gauss-Newton iterations. Defaults to 10.
    damping : float
        Gauss-Newton step damping factor in (0, 1]. Defaults to 0.7.
    anchors : tuple[int, ...]
        Indices of cameras to anchor as fixed gauge reference frames. Defaults to (0,).

    Returns
    -------
    motors : [n_cams] Motor
        Optimized camera poses.
    points : [n_points] Point
        Reconstructed world landmarks.
    quadrics : [n_points] Quadric
        Fused perspective cone quadrics (3D Gaussian splat precision ellipsoids).
    pose_covariance : [n_cams] TwistMap
        Camera pose covariance operator on the twist Lie algebra.
    """
    motors = initial_motors

    for _ in range(iterations):
        scaled_cones = reweight_cones(cameras, motors, local_cones)
        points, _ = triangulate_cones(motors, scaled_cones)

        # Pull world landmarks into local frames and evaluate polar plane residuals:
        local_points = motors << points[:, None]             # [n_points, n_cams] Point
        res = scaled_cones(local_points)                     # [n_points, n_cams] Plane

        # Camera pose variation and landmark variation Jacobians:
        j_cam = -scaled_cones(Twist.commutator(local_points)) # [n_points, n_cams] Plane <- Twist
        j_pt = scaled_cones(motors << Point)                  # [n_points, n_cams] Plane <- Point

        # Accumulate Hessian blocks:
        h_cam = (j_cam.transpose()(j_cam)).sum(axis=0)       # [n_cams] Twist <- Twist
        h_pt = (j_pt.transpose()(j_pt)).sum(axis=1)          # [n_points] Point <- Point
        h_cross = j_cam.transpose()(j_pt)                    # [n_points, n_cams] Twist <- Point

        # Schur complement: fold landmark compliance into camera stiffness:
        compliance = h_cross(h_pt.pinv(rcond=1e-4)[:, None](h_cross.transpose()))  # [n_points, n_cams] Twist <- Twist
        h_reduced = h_cam - compliance.sum(axis=0)           # [n_cams] Twist <- Twist

        # Gauss-Newton step and pose covariance (anchor cameras to fix gauge freedom):
        rhs = -(j_cam.transpose()(res)).sum(axis=0)          # [n_cams] Twist
        step = h_reduced.lstsq(rhs, rcond=1e-4)              # [n_cams] Twist
        for a in anchors:
            step = step.at[a].set(step[a] * 0)
        motors = motors * (step * (0.5 * damping)).exp()     # [n_cams] Motor

    points, q_fused = triangulate_reweighted(cameras, motors, local_cones)
    pose_covariance = h_reduced.pinv(rcond=1e-4)
    for a in anchors:
        pose_covariance = pose_covariance.at[a].set(pose_covariance[a] * 0)
    return motors, points, q_fused, pose_covariance
