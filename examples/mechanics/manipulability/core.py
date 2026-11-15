"""Velocity and force ellipsoids at the gripper of an articulated arm as quadrics in projective 3D space.

Each joint turns about an axis line; carried into the world by its cumulative pivot motor, that axis is
the joint's twist per unit rate. Both manipulability ellipsoids come directly from those twists.

An axis's commutator with the gripper tip is the tip's velocity per unit joint rate, an ideal direction.
The tip velocities reachable with unit joint rates fill a dual quadric from planes to points, formed as
the sum of those directions' dyads about the tip.

Joining a joint twist with the tip yields the plane through the joint axis and the tip. A tip force
lying in that plane meets or runs parallel to the axis and produces zero torque on the joint, while
pairing a force across that plane reads the joint torque. The forces resisted by unit joint torques fill
a point quadric from points to planes, formed as the sum of those torque-free planes' dyads.

By virtual work, each ellipsoid is the polar of the other in the unit sphere centred on the tip. Where
the arm stretches nearly straight, the pitch velocities align, flattening the velocity ellipsoid across
the arm and elongating the force ellipsoid along it.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import stack
from examples.geometry.scenegraph import core as arm

ga = arm.ga
mv = arm.mv
Point, Plane, PointMap = arm.Point, arm.Plane, arm.PointMap
# A joint's rate of turning about its axis in space.
Twist = ga.gatype.bivector()                       # Bivector+
# A point quadric maps each point to its polar plane; a dual quadric maps each plane to its pole.
Quadric = ga.gatype((Plane, Point))                # Plane+ <- Point-
DualQuadric = ga.gatype((Point, Plane))            # Point- <- Plane+
# The plane at infinity and the three coordinate planes through the origin.
w = mv.w                                           # Plane+
coordinate_planes = stack([mv.x, mv.y, mv.z])      # [3] Plane+


# --- math ----------------------------------------------------------------------------------------
def arm_axes(joint_angles: tuple[float, float, float, float]) -> tuple[PointMap, Point, Twist]:
    """Body placement maps (`[5] Point- <- Point-`), gripper tip (`[] Point-`), and world joint twists (`[4] Twist+`)."""
    bodies, pivots = arm.robot_arm(joint_angles)
    wrist_pivot = pivots[-1]                                    # [] Motor+
    tip = wrist_pivot >> mv.point([0.0, 0.0, 0.3])              # [] Point-
    # The base turret yaws in xy while the shoulder, elbow, and wrist pitch in zx.
    joint_generators = stack([mv.xy, mv.zx, mv.zx, mv.zx])      # [4] Twist+
    axes = pivots >> joint_generators                           # [4] Twist+
    return bodies, tip, axes


def velocity_ellipsoid(tip: Point, axes: Twist) -> DualQuadric:
    """Dual quadric (`[] Point- <- Plane+`) of tip velocities reachable with unit joint rates."""
    # Each joint's commutator with the tip gives the ideal velocity direction per unit joint rate.
    velocities = axes.commutator(tip)                           # [...] Point-
    return (velocities * (Plane & velocities)).sum() - tip * (Plane & tip)


def force_ellipsoid(tip: Point, axes: Twist) -> Quadric:
    """Point quadric (`[] Plane+ <- Point-`) of tip forces resisted by unit joint torques."""
    # Joining each joint twist with the tip gives the plane of tip forces that produce zero torque on that joint.
    zero_torque_planes = axes & tip                             # [...] Plane+
    return (zero_torque_planes * (zero_torque_planes & Point)).sum() - w * (w & Point)


def unit_sphere(centre: Point) -> Quadric:
    """Unit sphere (`[] Plane+ <- Point-`) centred on `centre` as the sum of coordinate-plane dyads through it."""
    central_planes = coordinate_planes - w * (coordinate_planes & centre)   # [3] Plane+
    return (central_planes * (central_planes & Point)).sum() - w * (w & Point)


def pose_ellipsoids(
    joint_angles: tuple[float, float, float, float],
    velocity_scale: float,
    force_scale: float,
) -> tuple[PointMap, Point, Quadric, Quadric]:
    """Body maps (`[5] Point- <- Point-`), tip (`[] Point-`), and scaled velocity and force quadrics (`[] Plane+ <- Point-`)."""
    bodies, tip, axes = arm_axes(joint_angles)
    # Invert the dual velocity quadric into a point quadric so both ellipsoids ray-trace as point surfaces.
    velocity = velocity_ellipsoid(tip, axes * velocity_scale).inverse()     # [] Plane+ <- Point-
    force = force_ellipsoid(tip, axes * force_scale)                        # [] Plane+ <- Point-
    return bodies, tip, velocity, force


def joint_loop_poses(
    n_frames: int,
    loop_centres: np.ndarray,
    loop_amplitudes: np.ndarray,
    loop_frequencies: np.ndarray,
    loop_phases: np.ndarray,
    velocity_scale: float,
    force_scale: float,
) -> Iterator[tuple[PointMap, Point, Quadric, Quadric]]:
    """Yields the arm's body maps, tip, and scaled velocity and force quadrics across `n_frames` poses of a closed joint loop."""
    for phase in np.linspace(0.0, 2 * np.pi, n_frames, endpoint=False):
        angles = loop_centres + loop_amplitudes * np.sin(loop_frequencies * phase + loop_phases)
        yield pose_ellipsoids(tuple(angles), velocity_scale, force_scale)
