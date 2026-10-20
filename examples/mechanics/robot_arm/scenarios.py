"""Scenes for the robot arm: statics at a rest pose, homing onto a target, and tracking a loop."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import stack
from examples.mechanics.robot_arm import core
from examples.mechanics.robot_arm.core import Point, Scalar, mv

# The elbow's and wrist's heights in the home pose, and the tip's; the rest angles of the base yaw and the
# two pitches; the joint rates and the tip force, its point and its direction, of the statics.
ELBOW_HEIGHT, WRIST_HEIGHT, TIP_HEIGHT = 1.0, 2.0, 3.0
REST_ANGLES = np.array([0.3, 0.5, 0.8])
RATES = np.array([0.5, -1.0, 0.25])
LOAD_AT, LOAD = [1.0, 0.0, 3.0], [0.0, 2.0, -1.0]
# The loop the tip follows: its centre, its radii along x and y and its height's swing, one target per frame.
LOOP_CENTRE = np.array([1.0, 1.0, 1.2])
LOOP_RADII = np.array([0.5, 0.5, 0.3])
N_FRAMES = 72
# The solver's steps when homing and per target when tracking; the links' half-width and half-depth; and
# the duration of a frame in milliseconds.
N_HOMING_STEPS, N_TRACKING_STEPS = 8, 2
LINK_WIDTH, LINK_DEPTH = 0.15, 0.06
FRAME_DURATION = 50
# The step of the finite difference the tip velocity is checked against.
DIFFERENCE = 1e-6


# --- math -----------------------------------------------------------------------------
def arm():
    """The joint axes in the home pose, the tip in the home pose, and a rest configuration."""
    # Joint axes in the home pose: yaw about z at the base, then two pitch joints about y, at the elbow
    # and the wrist. Each axis is a line to turn about: the meet of two planes.
    axis = stack([mv.y ^ mv.x, mv.x ^ (mv.z - ELBOW_HEIGHT * mv.w), mv.x ^ (mv.z - WRIST_HEIGHT * mv.w)])   # [joints] Twist
    tip_home = mv.point([0.0, 0.0, TIP_HEIGHT])                                # [] Point
    # A joint's state: its axis times its angle.
    return axis, tip_home, axis * REST_ANGLES                                  # [joints] Twist, [] Point, [joints] Twist


def loop_targets() -> Point:
    """Targets along a tilted loop in front of the arm."""
    phase = np.linspace(0.0, 2 * np.pi, N_FRAMES, endpoint=False)
    return mv.point(LOOP_CENTRE + LOOP_RADII * np.stack([np.sin(phase), np.cos(phase), np.sin(2 * phase)], axis=-1))


def link_boxes(n_links: int, width: float, depth: float) -> Point:
    """Corner points of slender boxes along z, one per unit link, in the home pose: shape (links, 8)."""
    corners = np.array([[x, y, z] for x in (-width, width) for y in (-depth, depth) for z in (0.0, 1.0)])
    return mv.point(corners[None] + np.array([[0.0, 0.0, 1.0]]) * np.arange(n_links)[:, None, None])


def statics() -> tuple[Point, Scalar]:
    """Tip velocity under joint rates, and the joint torques of a forque at the tip, at rest."""
    axis, tip_home, rest = arm()
    rates = axis * RATES                                                       # [joints] Twist
    # A force at a point is a forque: the point joined with the weighted direction.
    forque = mv.point(LOAD_AT) & mv.direction(LOAD)                            # [] Forque
    return core.mechanics(rest, axis, rates, tip_home, forque)


def homing() -> tuple[Scalar, Scalar]:
    """Solve from rest onto the start of the loop: the remaining tip error and the joint angles."""
    axis, tip_home, rest = arm()
    target = loop_targets()[0]
    joints, _, tip = core.inverse_kinematics(rest, axis, tip_home, target, N_HOMING_STEPS)
    # The distance, the norm of the join, and each angle, the joint projected on its axis.
    return (target & tip).norm(), (joints | axis) / (axis | axis)


def tracking() -> Iterator[tuple]:
    """Home onto the loop, then follow it: per target the link boxes, the target and the tip."""
    axis, tip_home, rest = arm()
    targets = loop_targets()
    boxes_home = link_boxes(len(REST_ANGLES), LINK_WIDTH, LINK_DEPTH)
    joints, _, _ = core.inverse_kinematics(rest, axis, tip_home, targets[0], N_HOMING_STEPS)
    for _, link_motors, target, tip in core.track(joints, axis, tip_home, targets, N_TRACKING_STEPS):
        yield link_motors[:, None] >> boxes_home, target, tip


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.animation import save_animation
    from examples.mechanics.robot_arm import render

    save_animation(render.animate_tracking(tracking()), "robot_arm", FRAME_DURATION)

    # checks
    # The tip velocity is the derivative of forward kinematics along the joint rates; homing reaches the
    # target.
    velocity, _ = statics()
    axis, tip_home, rest = arm()
    ahead, _ = core.forward_kinematics(rest + axis * RATES * DIFFERENCE)
    here, _ = core.forward_kinematics(rest)
    predicted = (here >> tip_home) + velocity * DIFFERENCE
    assert ((ahead >> tip_home) & predicted).norm().to_array().max() < 1e-9
    error, _ = homing()
    assert error.to_array().max() < 1e-8


if __name__ == "__main__":
    main()
