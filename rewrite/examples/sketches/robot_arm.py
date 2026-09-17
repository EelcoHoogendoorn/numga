"""A three-joint arm in PGA3D: joints, rates and the Jacobian are all lines.

A joint's configuration is its axis line scaled by the angle, a bivector, and so is its
rate. Forward kinematics exponentiates those bivectors in order; the frame in which each
joint acts is the product of the joints before it. Carrying the axis lines into those frames
gives the Jacobian, so it is never derived. The map from a twist to the tip velocity is the
commutator with the tip and a bivector hole, and the generalized force a forque exerts on a
joint is the pairing of the carried axis with that forque. No transpose is ever written.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from numga import Extensor, NumpyContext
from numga.algebras import PGA3D

from examples import PLOT_DIR
from examples.animation import capture, save_gif

ga = PGA3D
ctx = NumpyContext(ga)
mv = ctx.multivector
Point = ga.gatype.antivector()
Line = ga.gatype.bivector()
Motor = ga.gatype.rotor()
Scalar = ga.gatype.scalar()


# --- plumbing -------------------------------------------------------------------------
def point(coords: np.ndarray) -> Point:
    return mv.antivector(np.concatenate([coords, np.ones_like(coords[..., :1])], axis=-1))


def direction(coords: np.ndarray) -> Point:
    return mv.antivector(np.concatenate([coords, np.zeros_like(coords[..., :1])], axis=-1))


def euclidean(points: Point) -> np.ndarray:
    k = points.cast(ga.subspace.antivector()).kernel
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


def least_squares(columns: Point, delta: Point) -> Scalar:
    """The scalars, one per column, whose weighted sum of the columns best matches delta."""
    rhs = delta.cast(columns.output_subspace).kernel
    return mv.scalar(np.linalg.lstsq(columns.kernel.T, rhs, rcond=None)[0][:, None])


def draw_tracking(states, animation_path: str) -> None:
    """Render the link geometry and tip trail after solving the tracking motion."""
    if animation_path:
        fig = plt.figure(figsize=(5, 5), dpi=100)
        ax = fig.add_subplot(projection="3d")
        frames_out, trail = [], []
        for boxes, target, tip in states:
            trail.append(euclidean(tip))
            draw_arm(ax, boxes, target, trail)
            frames_out.append(capture(fig))
        plt.close(fig)
        save_gif(frames_out, animation_path, duration_ms=50)


# --- math -----------------------------------------------------------------------------
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
    pose, frames = mv.rotor(), []
    for step in (rest * 0.5).exp():
        frames.append(pose)
        pose = pose * step
    frames = Extensor.stack(frames)
    tip = pose >> tip_home                                         # the tip carried by the pose

    # 1. Velocities. Joint rates are bivectors along the axes; carried into their frames and
    #    summed they are the world twist, and the tip velocity is its commutator with the tip.
    rates = Extensor.stack([yaw * 0.5, pitch_1 * -1.0, pitch_2 * 0.25])   # joint rates: axis times angular rate
    twist = (frames >> rates).sum(axis=0)                          # each rate carried into its frame, summed
    velocity = twist.commutator(tip)                               # how the tip moves under the twist

    # 2. Statics. A force applied at a point is a forque: the join of the point with the
    #    weighted direction. Pairing a joint's carried axis with the forque gives a scalar, the
    #    work the load does per unit joint rate: the torque on that revolute joint.
    forque = point(np.array([1.0, 0.0, 3.0])) & direction(np.array([0.0, 2.0, -1.0]))
    joint_torques = (frames >> axis) & forque                      # carried axes paired with the load

    # 3. Inverse kinematics. The commutator of each carried axis with the tip is the tip
    #    velocity per unit rate of that joint; the least-squares weights of those velocities
    #    are the joint steps along the axes. The remaining error is the length of the join of
    #    tip and target.
    target = point(np.array([1.0, 1.5, 1.2]))
    joints = rest
    for iteration in range(8):
        pose, frames = mv.rotor(), []
        for step in (joints * 0.5).exp():
            frames.append(pose)
            pose = pose * step
        frames = Extensor.stack(frames)
        tip = pose >> tip_home
        columns = (frames >> axis).commutator(tip)                 # tip velocity per unit rate, per joint
        joints = joints + axis * least_squares(columns, target - tip)   # Newton step along the axes
        error = (target & tip).norm()                              # distance: the norm of the join
        print(f"iteration {iteration}: tip error {error.kernel[0]:.2e}")
    print("joint torques:", np.round(joint_torques.kernel[:, 0], 4))
    print("final joint angles:", np.round(((joints | axis) / (axis | axis)).kernel[:, 0], 4))   # angle = joint projected on its axis

    # 4. Tracking. The target runs around a loop starting where the solver left the tip; two
    #    of the same steps per frame keep the tip on it. Each link's box is carried by the
    #    frame after its joint, so the motors themselves are visible, not only the joint points.
    t = np.linspace(0.0, 2 * np.pi, 72, endpoint=False)
    targets = point(np.stack([1.0 + 0.5 * np.sin(t), 1.0 + 0.5 * np.cos(t), 1.2 + 0.3 * np.sin(2 * t)], axis=-1))
    states = []
    for target in targets:
        for _ in range(2):
            pose, frames = mv.rotor(), []
            for step in (joints * 0.5).exp():
                frames.append(pose)
                pose = pose * step
            frames = Extensor.stack(frames)
            tip = pose >> tip_home
            joints = joints + axis * least_squares((frames >> axis).commutator(tip), target - tip)
        link_motors = Extensor.concatenate([frames[1:], pose.reshape(1)])   # frame after each joint
        states.append((link_motors.reshape(3, 1) >> boxes_home, target, tip))
    draw_tracking(states, animation_path)

    # --- checks: kernel-level assertions, deliberately outside the demonstration ----------
    steps = (Extensor.stack((rest + rates * 1e-6, rest)) * 0.5).exp()
    poses = mv.rotor()
    for index in range(3):
        poses = poses * steps[:, index]
    finite = (poses[0] >> tip_home) - (poses[1] >> tip_home)
    np.testing.assert_allclose((velocity * 1e-6 - finite).kernel, 0.0, atol=1e-10)
    assert error.kernel[0] < 1e-8


if __name__ == "__main__":
    main()
