r"""N-camera multi-view reconstruction and bundle adjustment with perspective cone quadrics.

Each camera is a projective map on points, `Camera: Point <- Point`.

A quadric is written as a polarity map, `Plane <- Point`: a point's polar plane. Incidence of
a plane with a point is the plane-first join, `plane & point`, and a quadric's value at a
point is the join of the point's polar with the point, `quadric(p) & p`. That order is a
convention: the regressive product of a plane and a point changes sign with the dimension,
so the plane always comes first.

A pixel measurement on the sensor has a precision disc, a quadric on sensor points. Feeding
the camera into the disc and carrying its polar planes back through the camera lifts it into
a sight cone on scene points, whose cross-section widens with depth:
    cone = on_planes(camera)(disc(camera))              # Plane <- Point
on_planes(T) is the map on planes induced by a map on points, solved from the incidence
pairing; it satisfies on_planes(T)(l) & p == l & T(p) for every plane and point, singular T
included. A cone moves between frames like any map, `pose >> cone(pose << Point)`.

Bundle adjustment works on the cone quadrics directly:
1. Triangulate by summing each point's cones over its cameras. The fused cone's polar of its
   own vertex vanishes; a gauge dyad on the weight makes the vertex the pole of the plane at
   infinity, so `points = (fused + w * (w & Point)).solve(w).normalized()`.
2. Update poses by Newton's method on the cone value. The motion of a local point under a
   pose twist, its polar joined with itself, is the curvature; the point's polar joined with
   the motion is the gradient. The cone is the cost, so no residual metric is chosen.

This module contains the mathematics alone: GATypes and the geometric narrative
in one coherent scope.
"""

from __future__ import annotations

from examples.geometry.multiview.types import (
    Camera,
    Direction,
    Information,
    Motor,
    Plane,
    Point,
    Quadric,
    Scalar,
    Twist,
    w,
)


def on_planes(collineation: Camera):
    """The map on planes induced by a map on points, through incidence.

    Solving the incidence pairing against `Plane & collineation` gives the plane map with
    `on_planes(T)(l) & p == l & T(p)` for every plane l and point p, whether or not T is
    invertible. It carries a quadric's polar planes back through T.
    """
    return (Plane & Point).solve(Plane & collineation)       # Plane <- Plane


def triangulate_cones(
    motors: Motor,
    cones: Quadric,
) -> tuple[Point, Quadric]:
    """Triangulate scene points as the vertices of fused perspective cone quadrics.

    Parameters
    ----------
    motors : [n_cams] Motor
        Camera poses in world frame.
    cones : [n_points, n_cams] Quadric
        Perspective cone quadrics in each camera local frame.

    Returns
    -------
    points : [n_points] Point
        Reconstructed scene points (fused quadric vertices).
    fused : [n_points] Quadric
        Fused perspective cone quadrics (Gaussian splat precision ellipsoids).
    """
    # Move local cones to the world frame and sum them: quadratic constraints add.
    world_cones = motors >> cones(motors << Point)            # [n_points, n_cams] Plane <- Point
    fused = world_cones.sum(axis=-1)                          # [n_points] Plane <- Point

    # The fused cone's polar of its vertex vanishes. Adding the gauge dyad on the weight makes
    # that vertex the pole of the plane at infinity, without altering the spatial gradient:
    points = (fused + w * (w & Point)).solve(w).normalized()  # [n_points] Point
    return points, fused


def bundle_adjust(
    initial_motors: Motor,
    local_cones: Quadric,
    iterations: int,
    damping: float = 0.9,
    anchors: tuple[int, ...] = (0,),
) -> tuple[Motor, Point, Quadric]:
    """Jointly optimize camera poses and points purely via perspective cone quadrics.

    Parameters
    ----------
    initial_motors : [n_cams] Motor
        Initial camera pose estimates.
    local_cones : [n_points, n_cams] Quadric
        Perspective cone quadrics in each camera local frame.
    iterations : int
        Number of alternating Newton iterations.
    damping : float
        Step damping factor. Defaults to 0.9.
    anchors : tuple[int, ...]
        Indices of cameras to anchor as fixed gauge reference frames. Defaults to (0,).

    Returns
    -------
    motors : [n_cams] Motor
        Optimized camera poses.
    points : [n_points] Point
        Reconstructed scene points.
    fused : [n_points] Quadric
        Fused perspective cone quadrics.
    """
    motors = initial_motors

    for _ in range(iterations):
        # Triangulate scene points as the fused quadrics' vertices:
        points, _ = triangulate_cones(motors, local_cones)

        # Newton on the cone value over each camera's pose twist. A right perturbation of a pose
        # moves its local points by minus the commutator with the twist; the moved point's polar
        # joined with the motion is the curvature, the point's polar joined with the motion the gradient:
        local_points = motors << points[:, None]                  # [n_points, n_cams] Point
        motion = -Twist.commutator(local_points)                  # [n_points, n_cams] Point <- Twist: where each local point goes per unit right step of its camera's pose
        curvature = (local_cones(motion) & motion).sum(axis=0)    # [n_cams] Scalar <- (Twist, Twist)
        gradient = (local_cones(local_points) & motion).sum(axis=0)   # [n_cams] Scalar <- Twist

        # Solve the twist steps and anchor reference cameras to fix gauge freedom:
        step = curvature.lstsq(-gradient, rcond=1e-4)           # [n_cams] Twist
        for a in anchors:
            step = step.at[a].set(step[a] * 0)
        motors = motors * (step * (0.5 * damping)).exp()        # [n_cams] Motor

    points, fused = triangulate_cones(motors, local_cones)
    return motors, points, fused


def depths(cameras: Camera, motors: Motor, points: Point) -> Scalar:
    """Perpendicular depth of points from each camera: the weight of the projected point.

    Parameters
    ----------
    cameras : [n_cams] Camera
        Projective camera maps in local frame.
    motors : [n_cams] Motor
        Camera poses in world frame.
    points : [n_points] Point
        Scene points in world frame.

    Returns
    -------
    z : [n_points, n_cams] Scalar
        Positive perpendicular depth along the optical axis (clamped >= 0.1).
    """
    local_points = motors << points[:, None]                  # [n_points, n_cams] Point
    return (w & cameras(local_points)).abs().clip(0.1, None)  # [n_points, n_cams] Scalar


def reweight_cones(
    cameras: Camera,
    motors: Motor,
    cones: Quadric,
    iterations: int = 3,
) -> Quadric:
    """Reweight perspective cone quadrics into pixel units via Sampson depth scaling.

    Starts from algebraic cone quadrics and iteratively scales each cone by
    1 / z^2 using point depths, converging to true inverse pixel variance units.
    """
    weighted_cones = cones
    for _ in range(iterations):
        points, _ = triangulate_cones(motors, weighted_cones)
        z = depths(cameras, motors, points)
        weighted_cones = cones / (z ** 2)
    return weighted_cones


def bundle_adjust_schur(
    cameras: Camera,
    initial_motors: Motor,
    local_cones: Quadric,
    iterations: int,
    damping: float = 0.7,
    anchors: tuple[int, ...] = (0,),
) -> tuple[Motor, Point, Quadric, Information]:
    """Jointly optimize camera poses with Sampson depth reweighting and a Schur complement.

    Unlike the alternating solver, the joint Newton step accounts for the points moving with
    the cameras: the points' curvature and the cross term between points and cameras are
    folded into the cameras' curvature. The result is a true Newton step on the reduced
    problem, and the reduced curvature is the marginal information on each camera's pose.

    Parameters
    ----------
    cameras : [n_cams] Camera
        Projective camera maps in local frame.
    initial_motors : [n_cams] Motor
        Initial camera pose estimates.
    local_cones : [n_points, n_cams] Quadric
        Perspective cone quadrics in each camera local frame.
    iterations : int
        Number of Schur Newton iterations.
    damping : float
        Step damping factor. Defaults to 0.7.
    anchors : tuple[int, ...]
        Indices of cameras to anchor as fixed gauge reference frames. Defaults to (0,).

    Returns
    -------
    motors : [n_cams] Motor
        Optimized camera poses.
    points : [n_points] Point
        Reconstructed scene points.
    fused : [n_points] Quadric
        Fused perspective cone quadrics (Gaussian splat precision ellipsoids).
    information : [n_cams] Information
        Marginal curvature of the cost over each camera's pose twist; its inverse on
        readouts is the pose covariance.
    """
    motors = initial_motors

    for _ in range(iterations):
        scaled_cones = reweight_cones(cameras, motors, local_cones)
        points, _ = triangulate_cones(motors, scaled_cones)

        # Curvature blocks over camera twists, over point directions, and across the two. Cameras
        # move by twists and points by directions, ideal points, so neither block ever sees a
        # point's homogeneous scale and no gauge is needed:
        local_points = motors << points[:, None]                  # [n_points, n_cams] Point
        motion = -Twist.commutator(local_points)                  # [n_points, n_cams] Point <- Twist: where each local point goes per unit right step of its camera's pose
        moved = motors << Direction                               # [n_cams] Direction <- Direction: a world displacement into each camera
        h_cam = (scaled_cones(motion) & motion).sum(axis=0)       # [n_cams] Scalar <- (Twist, Twist)
        h_pt = (scaled_cones(moved) & moved).sum(axis=1)          # [n_points] Scalar <- (Direction, Direction)
        h_cross = scaled_cones(motion) & moved                    # [n_points, n_cams] Scalar <- (Twist, Direction)

        # Schur complement: fold the points' compliance into the cameras' curvature. Solving the
        # point curvature against the cross term, with the twist slot carried, gives each point's
        # displacement in response to a camera step; the moved point's polar joined with that
        # response, pulled into the camera, is the compliance:
        response = h_pt[:, None].solve(h_cross)                   # [n_points, n_cams] Direction <- Twist
        compliance = scaled_cones(motion) & (motors << response)  # [n_points, n_cams] Scalar <- (Twist, Twist)
        information = h_cam - compliance.sum(axis=0)              # [n_cams] Scalar <- (Twist, Twist)

        # Newton step on the reduced curvature; anchor cameras to fix gauge freedom:
        gradient = (scaled_cones(local_points) & motion).sum(axis=0)   # [n_cams] Scalar <- Twist
        step = information.lstsq(-gradient, rcond=1e-4)           # [n_cams] Twist
        for a in anchors:
            step = step.at[a].set(step[a] * 0)
        motors = motors * (step * (0.5 * damping)).exp()          # [n_cams] Motor

    points, fused = triangulate_cones(motors, reweight_cones(cameras, motors, local_cones))
    for a in anchors:
        information = information.at[a].set(information[a] * 0)
    return motors, points, fused, information

