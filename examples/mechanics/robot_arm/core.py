"""Kinematics, statics, and inverse-kinematics tracking for an articulated arm in 3D projective geometric algebra.

Each joint axis is a line bivector met from two planes, and scaling that axis by a joint angle or angular rate forms a twist.
Exponentiating the half-angle twists in sequence composes the joint motors in the frames accumulated from earlier joints, and carrying the home-pose axes into those frames gives the world twist for tip velocity and the pairing lines for static joint torques.
"""

from __future__ import annotations

from collections.abc import Iterator

from numga import NumpyContext, concatenate
from numga.algebras import PGA3D

ga = PGA3D
mv = NumpyContext(ga).multivector
Point = ga.gatype.antivector()                      # Antivector-
# A joint axis scaled by an angle or rate forms an extrinsic twist, while a line-bound load forms an intrinsic forque.
Twist = ga.gatype.bivector()                        # Bivector+
Forque = ga.gatype.antibivector()                   # Antibivector-
Motor = ga.gatype.rotor()                           # Rotor+
Scalar = ga.gatype.scalar()                         # Scalar+


# --- math -----------------------------------------------------------------------------
def forward_kinematics(joints: Twist) -> tuple[Motor, Motor]:
    """Right-multiplication of the joint motors (`[...] Motor+`) gives the tip motor (`[] Motor+`) and pre-joint frames (`[...] Motor+`)."""
    steps = (joints * 0.5).exp()                                               # [...] Motor+
    # Right-multiplying sequential joint steps equals reversing the cumulative product of their reverses.
    poses = steps.reverse().cumprod(axis=0).reverse()                          # [...] Motor+
    frames = concatenate([mv.rotor().reshape(1), poses[:-1]])                  # [...] Motor+
    return poses[-1], frames


def mechanics(
    joints: Twist,
    axes: Twist,
    rates: Twist,
    tip_home: Point,
    forque: Forque,
) -> tuple[Point, Scalar]:
    """Carried joint rates and axes determine the tip velocity (`[] Point-`) and static joint torques (`[...] Scalar+`) at `joints`."""
    pose, frames = forward_kinematics(joints)                                  # [] Motor+, [...] Motor+
    tip = pose >> tip_home                                                     # [] Point-

    # Carrying each joint rate into its frame and summing forms the spatial twist that moves the tip.
    twist = (frames >> rates).sum(axis=0)                                      # [] Twist+
    velocity = twist.commutator(tip)                                           # [] Point-

    # Pairing each carried joint axis with the applied forque reads the work per unit joint angle.
    joint_torques = (frames >> axes) & forque                                  # [...] Scalar+
    return velocity, joint_torques


def inverse_kinematics(
    joints: Twist,
    axes: Twist,
    tip_home: Point,
    target: Point,
    n_steps: int,
) -> tuple[Twist, Motor, Point]:
    """Gauss-Newton updates (`n_steps`) drive `joints` (`[...] Twist+`) toward `target`, returning the solved joints, link motors, and tip."""
    for _ in range(n_steps):
        pose, frames = forward_kinematics(joints)                              # [] Motor+, [...] Motor+
        tip = pose >> tip_home                                                 # [] Point-
        # Unit rates along the carried joint axes induce the basis tip velocities for the linear step.
        columns = (frames >> axes).commutator(tip)                             # [...] Point-
        joints = joints + axes * columns.lstsq(target - tip)                   # [...] Twist+
    pose, frames = forward_kinematics(joints)                                  # [] Motor+, [...] Motor+
    link_motors = concatenate([frames[1:], pose.reshape(1)])                   # [...] Motor+
    return joints, link_motors, pose >> tip_home


def tracking_states(
    joints: Twist,
    axes: Twist,
    tip_home: Point,
    targets: Point,
    n_steps: int,
) -> Iterator[tuple[Twist, Motor, Point, Point]]:
    """Yields the solved joints, link motors, target, and tip across a sequence of `targets` (`[...] Point-`)."""
    for target in targets:
        joints, link_motors, tip = inverse_kinematics(joints, axes, tip_home, target, n_steps)
        yield joints, link_motors, target, tip
