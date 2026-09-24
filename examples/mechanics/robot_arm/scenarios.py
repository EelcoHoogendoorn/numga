"""Scenes for the robot arm: statics at a rest pose, homing onto a target, and tracking a loop."""

from __future__ import annotations

import numpy as np

from numga import Extensor
from examples.mechanics.robot_arm import core
from examples.mechanics.robot_arm.core import Point, Scalar, direction, point

# Parameter along the loop the tip follows, one target per frame.
LOOP = np.linspace(0.0, 2 * np.pi, 72, endpoint=False)


def arm():
    """The joint axes in the home pose, the tip in the home pose, and a rest configuration."""
    # Joint axes in the home pose: yaw about z at the base, then two pitch joints about y.
    origin = point(np.zeros(3))
    elbow = point(np.array([0.0, 0.0, 1.0]))
    wrist = point(np.array([0.0, 0.0, 2.0]))
    yaw = origin & direction(np.array([0.0, 0.0, 1.0]))          # a line: the join of a point with a direction
    pitch_1 = elbow & direction(np.array([0.0, 1.0, 0.0]))
    pitch_2 = wrist & direction(np.array([0.0, 1.0, 0.0]))
    axis = Extensor.stack([yaw, pitch_1, pitch_2])           # one batch of three unit lines
    tip_home = point(np.array([0.0, 0.0, 3.0]))
    rest = axis * np.array([0.3, 0.5, 0.8])                       # joint state: axis times angle
    return axis, tip_home, rest


def loop_targets() -> Point:
    """Targets along a tilted loop in front of the arm."""
    return point(np.stack([1.0 + 0.5 * np.sin(LOOP), 1.0 + 0.5 * np.cos(LOOP), 1.2 + 0.3 * np.sin(2 * LOOP)], axis=-1))


def link_boxes(count: int, width: float, depth: float) -> Point:
    """Corner points of slender boxes along z, one per unit link, in the home pose: shape (count, 8)."""
    corners = np.array([[x, y, z] for x in (-width, width) for y in (-depth, depth) for z in (0.0, 1.0)])
    return point(corners[None] + np.array([[0.0, 0.0, 1.0]]) * np.arange(count)[:, None, None])


def statics() -> tuple[Point, Scalar]:
    """Tip velocity under joint rates, and the joint torques of a forque at the tip, at rest."""
    axis, tip_home, rest = arm()
    rates = axis * np.array([0.5, -1.0, 0.25])                    # joint rates: axis times angular rate
    # A force at a point is a forque: join that point with the weighted direction.
    forque = point(np.array([1.0, 0.0, 3.0])) & direction(np.array([0.0, 2.0, -1.0]))
    velocity, joint_torques = core.mechanics(rest, axis, rates, tip_home, forque)

    # --- checks
    # The tip velocity is the derivative of forward kinematics along the joint rates.
    step = 1e-6
    ahead, _ = core.forward_kinematics(rest + rates * step)
    here, _ = core.forward_kinematics(rest)
    predicted = (here >> tip_home) + velocity * step
    assert ((ahead >> tip_home) & predicted).norm().select[0].to_array().max() < 1e-9
    return velocity, joint_torques


def homing() -> tuple[Scalar, Scalar]:
    """Solve from rest onto the start of the loop: the remaining tip error and the joint angles."""
    axis, tip_home, rest = arm()
    target = loop_targets()[0]
    joints, _, tip = core.inverse_kinematics(rest, axis, tip_home, target, 8)
    error = (target & tip).norm().select[0]                      # distance: the norm of the join
    angles = (joints | axis) / (axis | axis)                      # angle = joint projected on its axis

    # --- checks
    assert error.to_array().max() < 1e-8
    return error, angles


def tracking():
    """Home onto the loop, then follow it with two steps per target: link boxes, target and tip."""
    axis, tip_home, rest = arm()
    targets = loop_targets()
    boxes_home = link_boxes(3, width=0.15, depth=0.06)
    joints, _, _ = core.inverse_kinematics(rest, axis, tip_home, targets[0], 8)
    for _, link_motors, target, tip in core.track(joints, axis, tip_home, targets, 2):
        yield link_motors[:, None] >> boxes_home, target, tip


if __name__ == "__main__":
    from examples.animation import save_animation
    from examples.mechanics.robot_arm import render

    velocity, joint_torques = statics()
    error, angles = homing()
    save_animation(render.animate_tracking(tracking()), "robot_arm", 50)

    print(f"homing tip error: {error.to_array().max():.2e}")
    print("joint torques:", np.round(joint_torques.to_array(), 4))
    print("homed joint angles:", np.round(angles.to_array(), 4))
