"""Scenarios for the robot arm covering rest-pose statics, homing, and closed-loop target tracking."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import stack
from examples.mechanics.robot_arm import core
from examples.mechanics.robot_arm.core import Point, Scalar, Twist, mv

elbow_height, wrist_height, tip_height = 1.0, 2.0, 3.0
rest_angles = np.array([0.3, 0.5, 0.8])
joint_rates = np.array([0.5, -1.0, 0.25])
load_point, load_force = [1, 0, 3], [0, 2, -1]
loop_centre = np.array([1.0, 1.0, 1.2])
loop_radii = np.array([0.5, 0.5, 0.3])
n_links = 3
n_frames = 90
n_homing_steps, n_tracking_steps = 8, 2
link_width, link_depth = 0.15, 0.06
frame_duration_ms = 40


# --- math -----------------------------------------------------------------------------
def arm() -> tuple[Twist, Point, Twist]:
    """Home-pose joint planes meet along three joint axes (`[3] Twist+`) with the home tip (`[] Point-`) and rest joints (`[3] Twist+`)."""
    # Base yaw turns about the vertical z-axis, while elbow and wrist pitch about horizontal lines at their heights.
    axes = stack([
        -mv.xy,
        mv.x ^ (mv.z - elbow_height * mv.w),
        mv.x ^ (mv.z - wrist_height * mv.w),
    ])                                                                         # [3] Twist+
    tip_home = mv.point([0, 0, tip_height])                                    # [] Point-
    return axes, tip_home, axes * rest_angles                                  # [3] Twist+, [] Point-, [3] Twist+


def loop_targets(n_frames: int) -> Point:
    """Uniformly spaced phases around a saddle curve place `n_frames` target points (`[n_frames] Point-`) in front of the arm."""
    phase = np.linspace(0.0, 2 * np.pi, n_frames, endpoint=False)
    offsets = loop_radii * np.stack([np.sin(phase), np.cos(phase), np.sin(2 * phase)], axis=-1)
    return mv.point(loop_centre + offsets)                                     # [n_frames] Point-


def link_boxes(n_links: int, width: float, depth: float) -> Point:
    """Unit-link box offsets stack into `n_links` home-pose link boxes (`[n_links, 8] Point-`)."""
    unit_corners = np.array([
        [across, deep, height]
        for across in (-width, width)
        for deep in (-depth, depth)
        for height in (0, 1)
    ])
    link_shifts = np.array([[0, 0, 1]]) * np.arange(n_links)[:, None, None]
    return mv.point(unit_corners[None] + link_shifts)                          # [n_links, 8] Point-


def statics() -> tuple[Point, Scalar]:
    """Rest-pose joint rates and a tip-link load produce the tip velocity (`[] Point-`) and static joint torques (`[3] Scalar+`)."""
    axes, tip_home, rest = arm()                                               # [3] Twist+, [] Point-, [3] Twist+
    rates = axes * joint_rates                                                 # [3] Twist+
    forque = mv.point(load_point) & mv.direction(load_force)                   # [] Forque-
    return core.mechanics(rest, axes, rates, tip_home, forque)


def homing() -> tuple[Scalar, Scalar]:
    """Gauss-Newton updates from rest onto the initial loop target leave a residual tip distance and three homed joint angles."""
    axes, tip_home, rest = arm()                                               # [3] Twist+, [] Point-, [3] Twist+
    target = loop_targets(n_frames)[0]                                         # [] Point-
    joints, _, tip = core.inverse_kinematics(rest, axes, tip_home, target, n_homing_steps)
    return (target & tip).norm(), (joints | axes) / (axes | axes)              # [] Scalar+, [3] Scalar+


def tracking(n_frames: int) -> Iterator[tuple[Point, Point, Point]]:
    """Yields the posed link boxes (`[3, 8] Point-`), target (`[] Point-`), and tip (`[] Point-`) across `n_frames` loop steps."""
    axes, tip_home, rest = arm()                                               # [3] Twist+, [] Point-, [3] Twist+
    targets = loop_targets(n_frames)                                           # [n_frames] Point-
    boxes_home = link_boxes(n_links, link_width, link_depth)                   # [n_links, 8] Point-
    homed_joints, _, _ = core.inverse_kinematics(rest, axes, tip_home, targets[0], n_homing_steps)
    for _, link_motors, target, tip in core.tracking_states(homed_joints, axes, tip_home, targets, n_tracking_steps):
        yield link_motors[:, None] >> boxes_home, target, tip


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    """Single-pass loop tracking writes `plots/robot_arm.gif` and verifies kinematics, virtual work, homing, and loop errors."""
    from examples.util.animation import save_animation
    from examples.mechanics.robot_arm import render

    tracked_states = []

    def recorded_states() -> Iterator[tuple[Point, Point, Point]]:
        for state in tracking(n_frames):
            tracked_states.append(state)
            yield state

    save_animation(render.tracking_frames(recorded_states()), "robot_arm", frame_duration_ms)

    # checks
    axes, tip_home, rest = arm()
    pose, frames = core.forward_kinematics(rest)
    assert (pose - (rest[0] * 0.5).exp() * (rest[1] * 0.5).exp() * (rest[2] * 0.5).exp()).scalar_norm_squared() < 1e-10
    assert ((mv.w & (pose >> tip_home)) - 1).abs() < 1e-8
    rates = axes * joint_rates
    forque = mv.point(load_point) & mv.direction(load_force)
    velocity, joint_torques = statics()
    assert (mv.w & velocity).abs() < 1e-8
    assert ((joint_torques * joint_rates).sum() - ((frames >> rates).sum(axis=0) & forque)).abs() < 1e-8
    targets = stack([step_target for _, step_target, _ in tracked_states])
    tracked_tips = stack([step_tip for _, _, step_tip in tracked_states])
    assert len(tracked_states) == n_frames
    assert (targets[0] & tracked_tips[0]).norm() < 1e-6
    assert ((targets & tracked_tips).norm() < 1e-4).all()


if __name__ == "__main__":
    main()
