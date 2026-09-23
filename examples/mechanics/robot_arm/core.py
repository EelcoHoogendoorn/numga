"""A three-joint arm in PGA3D: joints, rates and the Jacobian are all lines.

A joint's configuration is its axis line scaled by the angle, a bivector, and so is its
rate. Forward kinematics exponentiates those bivectors in order; the frame in which each
joint acts is the product of the joints before it. Carrying the axis lines into those frames
gives the Jacobian, so it is never derived. The map from a twist to the tip velocity is the
commutator with the tip and a bivector hole, and the generalized force a forque exerts on a
joint is the pairing of the carried axis with that forque. No transpose is ever written.
"""

from __future__ import annotations

import numpy as np

from numga import Extensor, NumpyContext
from numga.algebras import PGA3D

ga = PGA3D
ctx = NumpyContext(ga)
mv = ctx.multivector
Point = ga.gatype.antivector()
Line = ga.gatype.bivector()
Motor = ga.gatype.rotor()
Scalar = ga.gatype.scalar()


def point(coords: np.ndarray) -> Point:
    """Finite points at (..., 3) coordinates: the dual of the homogeneous vector."""
    return (mv("x y z", coords) + mv.w).dual()


def direction(coords: np.ndarray) -> Point:
    """Ideal points: the directions (..., 3), the dual of a weightless vector."""
    return mv("x y z", coords).dual()


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
                       iterations: int):
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
          iterations: int):
    """Follow targets, yielding joints, link poses, target and tip after each solve."""
    for target in targets:
        joints, link_motors, tip = inverse_kinematics(joints, axis, tip_home, target, iterations)
        yield joints, link_motors, target, tip
