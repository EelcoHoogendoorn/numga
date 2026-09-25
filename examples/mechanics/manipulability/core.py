"""Velocity and force ellipsoids at the gripper of an arm, as quadrics in PGA3D.

Each joint turns about an axis, a line; carried into the world by its pivot, the axis is the
joint's twist per unit rate. Both ellipsoids are built from those axes alone, independently.

Motion: an axis's commutator with the gripper tip is the tip's velocity per unit rate, a direction.
The tip velocities reachable with unit joint rates fill an ellipsoid; as a dual quadric, a map
from planes to points, it is the sum of those directions' dyads about the tip.

Incidence: an axis joined with the tip is the plane through them both. A force
at the tip within that plane meets the axis or runs parallel to it and loads the joint not at all;
across it, the joint's torque is the force read by that plane. The forces the joints resist with
unit torques fill an ellipsoid, a point quadric: the sum of those planes' dyads.

The two are reciprocal, each the other's polar in the unit sphere about the tip: the principle of
virtual work. Where the arm stretches straight two velocities line up; the velocity ellipsoid
flattens and the force ellipsoid runs off along the arm.
"""

from __future__ import annotations

import numpy as np

from numga import Extensor
from examples.geometry.scenegraph import core as arm

ga = arm.ga
mv = arm.mv
Point, Plane, Line = arm.Point, arm.Plane, arm.Line
# Points at infinity: velocities.
Direction = ga.gatype(ga.subspace.antivector().degenerate())
# A point quadric takes each point to its polar plane; a dual quadric takes each plane to its pole.
Quadric = ga.gatype((Plane, Point))                # Plane <- Point
DualQuadric = ga.gatype((Point, Plane))            # Point <- Plane
# The plane at infinity.
w = mv.w                                           # [] Plane
# The coordinate planes through the origin, normal to x, y and z.
coordinate_planes = mv("x y z", np.eye(3))         # [3] Plane


# --- math ----------------------------------------------------------------------------------------
def arm_axes(joint_angles: tuple) -> tuple[arm.PointMap, Point, Line]:
    """The arm's body maps, its gripper tip, and each joint's axis in the world.

    A joint's axis is its rotation generator, carried into the world by its pivot: the base yaws
    about z, the shoulder, elbow and wrist pitch about y. It is also the joint's twist per unit rate.
    """
    bodies, pivots = arm.robot_arm(joint_angles)
    # The top face of the gripper, on the wrist.
    tip = pivots[-1] >> arm.point(np.array([0.0, 0.0, 0.3]))     # [] Point
    axes = Extensor.stack([pivots[0] >> mv.xy] + [pivot >> mv.zx for pivot in pivots[1:]])    # [n_joints] Line
    return bodies, tip, axes


def velocity_ellipsoid(tip: Point, axes: Line) -> DualQuadric:
    """The tip velocities reachable with unit joint rates: the velocities' dyads about the tip.

    `tip` and the joint `axes` may be in any frame, the same one for both; the ellipsoid comes out
    in that frame.
    """
    # The tip velocity per unit rate of each joint.
    velocities = axes.commutator(tip).cast(Direction)            # [n_joints] Direction
    return (velocities * (Plane & velocities)).sum() - tip * (Plane & tip)


def force_ellipsoid(tip: Point, axes: Line) -> Quadric:
    """The tip forces resisted with unit joint torques: the dyads of the planes through each joint's
    axis and the tip, a point quadric.

    `tip` and the joint `axes` may be in any frame, the same one for both; the ellipsoid comes out
    in that frame.
    """
    # Each joint's axis joined with the tip.
    torque_free = axes & tip                                       # [n_joints] Plane
    return (torque_free * (torque_free & Point)).sum() - w * (w & Point)


def unit_sphere(centre: Point) -> Quadric:
    """The unit sphere about a point: the dyads of the coordinate planes through it."""
    through = coordinate_planes - w * (coordinate_planes & centre)  # [3] Plane
    return (through * (through & Point)).sum() - w * (w & Point)
