"""Extended Position-Based Dynamics (XPBD) and Geometric Constraint Projection in PGA.

In classical robotics and game physics engines, projecting a rigid-body spherical joint or
distance constraint requires deriving explicit Jacobian matrices, computing generalized
inverse-mass matrices J M^-1 J^T, solving for Lagrange multipliers, and applying separate
linear and angular impulses to update velocities and quaternion poses.

In 3D Projective Geometric Algebra (PGA = R_{3,0,1}) with Extensors, the entire constraint
projection is coordinate-free geometry:
1. World anchors:
       world_anchors = motors >> anchors
2. Constraint violation line (the join of two points):
       line = world_anchors[0] & world_anchors[1]
   The line's norm is the exact Euclidean distance; its direction is the wrench line of action!
3. Pullback to body frames:
       local_dir = motors << direction
4. Inertia maps wrench line to twist step:
       step = inertia_inv(local_dir)
5. Effective inertial compliance (generalized inverse mass):
       w = step & local_dir
6. XPBD Lagrange multiplier:
       delta_lambda = distance / (compliance / dt^2 + sum(w))
7. Corrective displacement via the Lie algebra exponential map:
       motors * (step * connectivity * delta_lambda * -0.5).exp()

A fixed body has infinite mass: its inverse inertia is zero, so no constraint and no
external forque ever moves it.

The context is chosen by the caller, so the same lines run on NumPy and under jax.jit.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from numga import Extensor
from numga.algebras import PGA3D
from examples.mechanics import lie_integrators as lie

ga = PGA3D
Scalar = ga.gatype.scalar()
Motor = ga.gatype.rotor()
Rate = ga.gatype.bivector()
Forque = ga.gatype.antibivector()
Inertia = ga.gatype((Forque, Rate))
InverseInertia = ga.gatype((Rate, Forque))
Point = ga.gatype.antivector()


class Chain(NamedTuple):
    """Batched state of the rigid bodies in the chain."""
    motor: Motor                  # [bodies] pose
    rate: Rate                    # [bodies] body-frame rate
    first_moment: Point           # [bodies] mass-weighted centre, in the body frame
    inertia: Inertia              # [bodies]
    inertia_inv: InverseInertia   # [bodies] zero for a fixed body
    damping: Scalar               # [bodies]
    gravity: Point                # [bodies] ideal point: the gravitational acceleration


class Joints(NamedTuple):
    """A disjoint set of pairwise joints that can be relaxed simultaneously."""
    bodies: np.ndarray            # [2, joints] indices of the joined bodies
    anchors: Point                # [2, joints] anchor points, each in its own body frame
    compliance: Scalar            # [joints] inverse stiffness alpha


# --- math -----------------------------------------------------------------------------
def project_distance_constraint(
    motors: Motor, anchors: Point, inertia_inv: InverseInertia, compliance: Scalar, dt: float,
) -> Motor:
    """Close the gap between the anchors of body pairs, motors and anchors of shape [2, ...]."""
    # 1. Transform anchor points to world frame via motor pushforward:
    world_anchors = motors >> anchors

    # 2. Join the two world points into a line:
    # In PGA, the regressive product (&) of two points is the line connecting them.
    # Its norm is the Euclidean distance between the anchors. A closed joint has a
    # vanishing line; the small offsets keep its direction and multiplier at zero.
    line = world_anchors[0] & world_anchors[1]
    magnitude = line.norm().select[0]
    direction = line / (magnitude + 1e-24)

    # 3. Pull the wrench line back into each body's local coordinate frame:
    local_dir = motors << direction

    # 4. Inverse inertia maps the wrench line to an impulsive twist step:
    steps = inertia_inv(local_dir)

    # 5. Generalized inverse mass / compliance along the constraint line:
    # Pairing the twist with the wrench line gives the effective compliance:
    # w = step & local_dir = I^-1(dir) & dir
    inertial_compliances = steps & local_dir
    total_compliance = (compliance / (dt**2)) + inertial_compliances.sum(axis=0) + 1e-24

    # 6. XPBD Lagrange multiplier with body signs (+1 for body 0, -1 for body 1):
    multiplier = (magnitude / total_compliance) * np.array([[+1], [-1]])

    # 7. Equal and opposite impulse steps on body 0 and body 1:
    impulse = steps * multiplier

    # 8. Corrective motor displacement via Lie algebra exponential map (half-angle):
    return motors * (impulse * -0.5).exp()


def project_velocity_constraint(
    motors: Motor, rates: Rate, anchors: Point, inertia_inv: InverseInertia,
) -> Rate:
    """Cancel the relative velocity of the anchors of body pairs, with an impulse along it."""
    # Relative anchor velocity in world coordinates:
    anchors_map = anchors & anchors.commutator(Rate)
    velocities = motors >> anchors_map(rates)
    forque = velocities[1] - velocities[0]
    magnitude = forque.norm().select[0]
    direction = forque / (magnitude + 1e-24)

    local_dir = motors << direction
    steps = inertia_inv(local_dir)
    inertial_compliances = steps & local_dir
    total_compliance = inertial_compliances.sum(axis=0) + 1e-24
    multiplier = (magnitude / total_compliance) * np.array([[+1], [-1]])
    impulse = steps * multiplier
    return rates + impulse


def external_forque(motor: Motor, rate: Rate, first_moment: Point, gravity: Point, damping: Scalar) -> Forque:
    """Compute external forque line in body-local frame (gravity + damping)."""
    # The weight is the line through the centre of mass along gravity: their join.
    gravity_local = motor << gravity
    grav_forque = first_moment & gravity_local
    damp_forque = -(rate * damping).dual()
    return grav_forque + damp_forque


def post_integrate(old_motor: Motor, relaxed_motor: Motor, dt: float) -> tuple[Motor, Rate]:
    """Verlet post-integration: update rates from relaxed motors."""
    motor_normalized = relaxed_motor.normalized()
    rate = (~old_motor * motor_normalized).log() * (-2.0 / dt)
    return motor_normalized, rate


def step(chain: Chain, partitions: list[Joints], dt: float) -> Chain:
    """Perform a single XPBD integration step on the chain."""
    def forque(motor: Motor, rate: Rate) -> Forque:
        return external_forque(motor, rate, chain.first_moment, chain.gravity, chain.damping)

    # 1. Unconstrained inertial pre-integration:
    motor, rate = lie.explicit_rk4(chain.motor, chain.rate, chain.inertia, chain.inertia_inv, dt, forque)

    # 2. Relax position constraints (Gauss-Seidel over red-black partitions):
    for joints in partitions:
        pair = joints.bodies
        relaxed = project_distance_constraint(
            motor[pair], joints.anchors, chain.inertia_inv[pair], joints.compliance, dt
        )
        motor = motor.at[pair].set(relaxed)

    # 3. Post-integration: recover rates from motor displacement:
    motor, rate = post_integrate(chain.motor, motor, dt)

    # 4. Resolve velocity constraints:
    for joints in partitions:
        pair = joints.bodies
        resolved = project_velocity_constraint(motor[pair], rate[pair], joints.anchors, chain.inertia_inv[pair])
        rate = rate.at[pair].set(resolved)

    return chain._replace(motor=motor, rate=rate)


def advance(chain: Chain, partitions: list[Joints], substeps: int, dt: float) -> Chain:
    """Advance the chain by dt in equal substeps."""
    for _ in range(substeps):
        chain = step(chain, partitions, dt / substeps)
    return chain


def joint_gaps(motor: Motor, partitions: list[Joints]) -> Scalar:
    """The Euclidean separation of the two anchors of every joint: the norm of their join."""
    def gap(joints: Joints) -> Scalar:
        world_anchors = motor[joints.bodies] >> joints.anchors
        # The join of two points is a simple line: its norm has no pseudoscalar part.
        return (world_anchors[0] & world_anchors[1]).norm().select[0]

    return Extensor.concatenate([gap(joints) for joints in partitions])
