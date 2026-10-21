"""A three-joint arm in PGA3D: joints, rates and the Jacobian are all twists, lines met from planes.

A joint's configuration is its axis line scaled by the angle, a bivector, and so is its
rate. Forward kinematics exponentiates those bivectors in order; the frame in which each
joint acts is the product of the joints before it. Carrying the axis lines into those frames
gives the Jacobian, so it is never derived. The map from a twist to the tip velocity is the
commutator of the twist with the tip, `Twist.commutator(tip)`, and the generalized force a forque exerts on a
joint is the pairing of the carried axis with that forque. No transpose is ever written.
"""

from __future__ import annotations

from numga import NumpyContext, concatenate
from numga.algebras import PGA3D

ga = PGA3D
ctx = NumpyContext(ga)
mv = ctx.multivector
Point = ga.gatype.antivector()
# A joint's axis scaled by an angle or a rate: a twist, the meet of two planes; and a force line, a forque.
Twist = ga.gatype.bivector()
Forque = ga.gatype.antibivector()
Motor = ga.gatype.rotor()
Scalar = ga.gatype.scalar()


# --- math -----------------------------------------------------------------------------
def forward_kinematics(joints: Twist) -> tuple[Motor, Motor]:
    """Return the tip pose and the frame before each joint's rotation."""
    steps = (joints * 0.5).exp()                                               # [joints] Motor
    # The steps multiply on the right: their running product is the reverse of the running product of
    # their reverses.
    poses = steps.reverse().cumprod(axis=0).reverse()                          # [joints] Motor
    return poses[-1], concatenate([mv.rotor().reshape(1), poses[:-1]])


def mechanics(joints: Twist, axis: Twist, rates: Twist, tip_home: Point,
              forque: Forque) -> tuple[Point, Scalar]:
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


def inverse_kinematics(joints: Twist, axis: Twist, tip_home: Point, target: Point,
                       iterations: int):
    """Solve towards a target; return updated joints, link poses and the tip."""
    for _ in range(iterations):
        pose, frames = forward_kinematics(joints)
        tip = pose >> tip_home
        # Tip velocity per unit rate, per joint.
        columns = (frames >> axis).commutator(tip)
        # Fit the desired tip displacement as a weighted sum of these velocities;
        # apply those weights as joint increments along the home-pose axes.
        joints = joints + axis * columns.lstsq(target - tip)
    pose, frames = forward_kinematics(joints)
    # Frame after each joint.
    link_motors = concatenate([frames[1:], pose.reshape(1)])
    return joints, link_motors, pose >> tip_home


def track(joints: Twist, axis: Twist, tip_home: Point, targets: Point,
          iterations: int):
    """Follow targets, yielding joints, link poses, target and tip after each solve."""
    for target in targets:
        joints, link_motors, tip = inverse_kinematics(joints, axis, tip_home, target, iterations)
        yield joints, link_motors, target, tip
