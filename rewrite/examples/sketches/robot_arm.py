"""A three-joint arm in PGA3D: joints, rates and the Jacobian are all lines.

A joint's configuration is its axis line scaled by the angle, a bivector, and so is its
rate. Forward kinematics exponentiates those bivectors in order; the frame in which each
joint acts is the product of the joints before it. Carrying the axis lines into those frames
gives the Jacobian, so it is never derived. The map from a twist to the tip velocity is the
commutator with the tip and a bivector hole, and the generalized force a forque exerts on a
joint is the pairing of the carried axis with that forque. No transpose is ever written.
"""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.pyplot as plt
import numpy as np

from numga import Extensor, NumpyContext
from numga.algebras import PGA3D

from examples import PLOT_DIR
from examples.animation import capture, save_gif

# --- scenario algebra -----------------------------------------------------------------
ga = PGA3D
ctx = NumpyContext(ga)
mv = ctx.multivector
Point = ga.gatype.antivector()
Line = ga.gatype.bivector()
Motor = ga.gatype.rotor()
Scalar = ga.gatype.scalar()


# --- math -----------------------------------------------------------------------------


def forward_kinematics(joints: Line) -> tuple[Motor, Motor]:
    """Return the tip pose and the frame before each joint's rotation."""
    pose, frames = mv.rotor(), []
    for step in (joints * 0.5).exp():
        frames.append(pose)
        pose = pose * step
    return pose, Extensor.stack(frames)


def mechanics(joints: Line, axis: Line, rates: Line, tip_home: Point,
              forque: Line) -> tuple[Point, Scalar]:
    """Compute tip velocity and joint torques at the supplied configuration."""
    pose, frames = forward_kinematics(joints)
    tip = pose >> tip_home

    # Carry each joint's rate into its frame and sum to get the world twist.
    # Its commutator with the tip gives the tip velocity.
    twist = (frames >> rates).sum(axis=0)
    velocity = twist.commutator(tip)

    # Pair each carried axis with the applied forque: work per unit joint rate.
    joint_torques = (frames >> axis) & forque
    return velocity, joint_torques


def inverse_kinematics(joints: Line, axis: Line, tip_home: Point, target: Point,
                       iterations: int) -> tuple[Line, Motor, Point]:
    """Solve towards a target; return updated joints, link poses and the tip."""
    for _ in range(iterations):
        pose, frames = forward_kinematics(joints)
        tip = pose >> tip_home
        columns = (frames >> axis).commutator(tip)  # Tip velocity per unit rate, per joint.
        # Fit the desired tip displacement as a weighted sum of these velocities;
        # apply those weights as joint increments along the original axes.
        joints = joints + axis * columns.lstsq(target - tip)
    pose, frames = forward_kinematics(joints)
    link_motors = Extensor.concatenate([frames[1:], pose.reshape(1)])  # Frame after each joint.
    return joints, link_motors, pose >> tip_home


def track(joints: Line, axis: Line, tip_home: Point, targets: Point,
          iterations: np.ndarray) -> Iterator[tuple[Line, Motor, Point, Point]]:
    """Follow targets, yielding joints, link poses, target and tip after each solve."""
    for target, count in zip(targets, iterations):
        joints, link_motors, tip = inverse_kinematics(joints, axis, tip_home, target, count)
        yield joints, link_motors, target, tip


# --- plotting -------------------------------------------------------------------------


def euclidean(points: Point) -> np.ndarray:
    k = points.cast(ga.subspace("yzw zxw xyw zyx")).kernel
    return k[..., :3] / k[..., 3:]


BOX_EDGES = [(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


def link_boxes(count: int, width: float, depth: float) -> Point:
    """Corner points of slender boxes along z, one per unit link, in the home pose: shape (count, 8)."""
    corners = np.array([[x, y, z] for x in (-width, width) for y in (-depth, depth) for z in (0.0, 1.0)])
    return point(corners[None] + np.array([[0.0, 0.0, 1.0]]) * np.arange(count)[:, None, None])


def draw_arm(ax, boxes: Point, target: Point, trail: list[np.ndarray]) -> None:
    """Draw the link boxes, the target, and the tip's trail so far."""
    ax.cla()
    for xyz, color in zip(euclidean(boxes), ("tab:blue", "tab:cyan", "tab:purple")):
        for a, b in BOX_EDGES:
            ax.plot(*zip(xyz[a], xyz[b]), color=color, linewidth=1.2)
    ax.scatter(*euclidean(target), color="tab:red", s=40)
    if trail:
        ax.plot(*np.array(trail).T, color="tab:red", linewidth=0.8, alpha=0.6)
    ax.set_xlim(-0.5, 2.0); ax.set_ylim(-1.25, 1.25); ax.set_zlim(0, 3.0); ax.set_box_aspect((1, 1, 1.2))
    ax.view_init(elev=22, azim=-50)


def draw_tracking(states, boxes_home: Point, animation_path: str) -> None:
    """Render the link geometry and tip trail after solving the tracking motion."""
    if animation_path:
        fig = plt.figure(figsize=(5, 5), dpi=100)
        ax = fig.add_subplot(projection="3d")
        frames_out, trail = [], []
        for _, link_motors, target, tip in states:
            boxes = link_motors[:, None] >> boxes_home
            trail.append(euclidean(tip))
            draw_arm(ax, boxes, target, trail)
            frames_out.append(capture(fig))
        plt.close(fig)
        save_gif(frames_out, animation_path, duration_ms=50)


# --- scenario -------------------------------------------------------------------------


def point(coords: np.ndarray) -> Point:
    return mv("yzw zxw xyw", coords) + mv.zyx


def direction(coords: np.ndarray) -> Point:
    return mv("yzw zxw xyw", coords)


def main(animation_path: str = str(PLOT_DIR / "sketch_robot_arm.gif")) -> None:
    # Joint axes in the home pose: yaw about z at the base, then two pitch joints about y.
    origin = point(np.zeros(3))
    elbow = point(np.array([0.0, 0.0, 1.0]))
    wrist = point(np.array([0.0, 0.0, 2.0]))
    yaw = origin & direction(np.array([0.0, 0.0, 1.0]))          # a line: the join of a point with a direction
    pitch_1 = elbow & direction(np.array([0.0, 1.0, 0.0]))
    pitch_2 = wrist & direction(np.array([0.0, 1.0, 0.0]))
    axis = Extensor.stack([yaw, pitch_1, pitch_2])                # one batch of three unit lines
    tip_home = point(np.array([0.0, 0.0, 3.0]))
    boxes_home = link_boxes(3, width=0.15, depth=0.06)

    rest = Extensor.stack([yaw * 0.3, pitch_1 * 0.5, pitch_2 * 0.8])   # joint state: axis times angle
    rates = Extensor.stack([yaw * 0.5, pitch_1 * -1.0, pitch_2 * 0.25])   # joint rates: axis times angular rate
    # A force at a point is a forque: join that point with the weighted direction.
    forque = point(np.array([1.0, 0.0, 3.0])) & direction(np.array([0.0, 2.0, -1.0]))

    # Home at the start of the path, then follow the loop with two steps per frame.
    t = np.linspace(0.0, 2 * np.pi, 72, endpoint=False)
    targets = point(np.stack([1.0 + 0.5 * np.sin(t), 1.0 + 0.5 * np.cos(t), 1.2 + 0.3 * np.sin(2 * t)], axis=-1))
    targets = Extensor.concatenate([targets[:1], targets])
    iterations = np.full(targets.shape, 2)
    iterations[0] = 8

    velocity, joint_torques = mechanics(rest, axis, rates, tip_home, forque)
    states = track(rest, axis, tip_home, targets, iterations)
    joints, _, target, tip = next(states)                           # the homed state
    error = (target & tip).norm()                                  # distance: the norm of the join
    angles = (joints | axis) / (axis | axis)                        # angle = joint projected on its axis
    draw_tracking(list(states), boxes_home, animation_path)

    # --- readout -----------------------------------------------------------------------
    print(f"homing tip error: {error.kernel[0]:.2e}")
    print("joint torques:", np.round(joint_torques.kernel[:, 0], 4))
    print("final joint angles:", np.round(angles.kernel[:, 0], 4))

    # --- checks ------------------------------------------------------------------------
    steps = (Extensor.stack((rest + rates * 1e-6, rest)) * 0.5).exp()
    poses = mv.rotor()
    for index in range(axis.shape[0]):
        poses = poses * steps[:, index]
    finite = (poses[0] >> tip_home) - (poses[1] >> tip_home)
    np.testing.assert_allclose((velocity * 1e-6 - finite).kernel, 0.0, atol=1e-10)
    assert error.kernel[0] < 1e-8


if __name__ == "__main__":
    main()
