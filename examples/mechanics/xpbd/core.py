"""Extended position-based dynamics of a chain of rigid links, in PGA3D.

Each step moves the links freely, then closes every joint they have pulled apart. A joint is two
anchor points, one on each link, that should coincide; the line joining them, their regressive
product, says both how far apart they are, its norm, and along which line the correcting load acts,
its direction, a forque. Pulled into each link's own frame, that forque turns through the link's
inverse inertia into the twist it would cause, and the twist paired with the forque is the link's
compliance along that line. One Lagrange multiplier, the gap over the summed compliances and the joint's
own, sets equal and opposite twists, and each link moves by their exponential. After the positions,
the joints' relative velocities are cancelled the same way, with impulses. A fixed link has zero
inverse inertia, so no load ever moves it.

The joints are relaxed in two alternating halves whose links do not overlap, so each half relaxes in
one batch. The context is chosen by the caller, so the same lines run on NumPy and under jax.jit.

In matrix notation the projection reads as explicit Jacobian matrices, a generalized inverse-mass
matrix built from the Jacobians and the inverse mass matrix, a solve for Lagrange multipliers, and
separate linear and angular impulses on velocities and quaternion poses.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from numga import concatenate
from numga.algebras import PGA3D
from examples.mechanics import lie_integrators as lie

ga = PGA3D
Scalar = ga.gatype.scalar()                         # Scalar+
Motor = ga.gatype.rotor()                           # Rotor+
Rate = ga.gatype.bivector()                         # Bivector+
Forque = ga.gatype.antibivector()                   # Antibivector-
Inertia = ga.gatype((Forque, Rate))
InverseInertia = ga.gatype((Rate, Forque))
Point = ga.gatype.antivector()                      # Antivector-


class Chain(NamedTuple):
    """Batched state of the rigid bodies in the chain."""
    motor: Motor                  # [bodies] Motor+
    # The rate in the body frame.
    rate: Rate                    # [bodies] Rate+
    # The mass-weighted centre, in the body frame.
    first_moment: Point           # [bodies] Point-
    inertia: Inertia              # [bodies] Forque- <- Rate+
    # Zero for a fixed body.
    inverse_inertia: InverseInertia   # [bodies] Rate+ <- Forque-
    damping: Scalar               # [bodies] Scalar+
    # An ideal point: the gravitational acceleration.
    gravity: Point                # [bodies] Point-


class Joints(NamedTuple):
    """A disjoint set of pairwise joints that can be relaxed simultaneously."""
    # Indices of the joined bodies.
    bodies: np.ndarray            # [ends, joints] int
    # Anchor points, each in its own body frame.
    anchors: Point                # [ends, joints] Point-
    # Inverse stiffness, alpha.
    compliance: Scalar            # [joints] Scalar+


# --- math -----------------------------------------------------------------------------
def project_distance_constraint(
    motors: Motor, anchors: Point, inverse_inertia: InverseInertia, compliance: Scalar, dt: float,
) -> Motor:
    """The motors of body pairs, `[2, ...]`, moved to close the gap between their anchors."""
    # 1. Transform anchor points to world frame via motor pushforward:
    world_anchors = motors >> anchors

    # 2. Join the two world points into a line:
    # In PGA, the regressive product (&) of two points is the line connecting them.
    # Its norm is the Euclidean distance between the anchors. A closed joint has a
    # vanishing line; the small offsets keep its direction and multiplier at zero.
    line = world_anchors[0] & world_anchors[1]
    magnitude = line.norm()
    direction = line / (magnitude + 1e-24)

    # 3. Pull the forque back into each body's local coordinate frame:
    local_direction = motors << direction

    # 4. Inverse inertia maps the forque to an impulsive twist step:
    steps = inverse_inertia(local_direction)

    # 5. Generalized inverse mass / compliance along the constraint line:
    # Pairing the twist with the forque gives the effective compliance:
    # `inverse_inertia(local_direction) & local_direction`
    inertial_compliances = steps & local_direction
    total_compliance = (compliance / (dt**2)) + inertial_compliances.sum(axis=0) + 1e-24

    # 6. XPBD Lagrange multiplier with body signs (+1 for body 0, -1 for body 1):
    multiplier = (magnitude / total_compliance) * np.array([[+1], [-1]])

    # 7. Equal and opposite impulse steps on body 0 and body 1:
    impulse = steps * multiplier

    # 8. Corrective motor displacement via Lie algebra exponential map (half-angle):
    return motors * (impulse * -0.5).exp()


def project_velocity_constraint(
    motors: Motor, rates: Rate, anchors: Point, inverse_inertia: InverseInertia,
) -> Rate:
    """The rates of body pairs, `[2, ...]`, with the relative velocity of their anchors cancelled by an impulse along
    it."""
    # Relative anchor velocity in world coordinates:
    anchors_map = anchors.regressive(anchors.commutator(Rate))
    velocities = motors >> anchors_map(rates)
    forque = velocities[1] - velocities[0]
    magnitude = forque.norm().select[0]
    direction = forque / (magnitude + 1e-24)

    local_direction = motors << direction
    steps = inverse_inertia(local_direction)
    inertial_compliances = steps & local_direction
    total_compliance = inertial_compliances.sum(axis=0) + 1e-24
    multiplier = (magnitude / total_compliance) * np.array([[+1], [-1]])
    impulse = steps * multiplier
    return rates + impulse


def external_forque(motor: Motor, rate: Rate, first_moment: Point, gravity: Point, damping: Scalar) -> Forque:
    """The external forque in the body frame: the weight, and the damping's drag."""
    # The weight is the line through the centre of mass along gravity: their join.
    gravity_local = motor << gravity
    weight = first_moment & gravity_local
    drag = -(rate * damping).dual()
    return weight + drag


def post_integrate(old_motor: Motor, relaxed_motor: Motor, dt: float) -> tuple[Motor, Rate]:
    """The relaxed motors, normalized, and the rates read back from their displacement: Verlet's post-integration."""
    motor_normalized = relaxed_motor.normalized()
    rate = (~old_motor * motor_normalized).log() * (-2.0 / dt)
    return motor_normalized, rate


def step(chain: Chain, partitions: list[Joints], dt: float) -> Chain:
    """The chain one XPBD step later."""
    def forque(motor: Motor, rate: Rate) -> Forque:
        return external_forque(motor, rate, chain.first_moment, chain.gravity, chain.damping)

    # 1. Unconstrained inertial pre-integration:
    motor, rate = lie.explicit_rk4(chain.motor, chain.rate, chain.inertia, chain.inverse_inertia, dt, forque)

    # 2. Relax position constraints (Gauss-Seidel over red-black partitions):
    for joints in partitions:
        pair = joints.bodies
        relaxed = project_distance_constraint(
            motor[pair], joints.anchors, chain.inverse_inertia[pair], joints.compliance, dt
        )
        motor = motor.at[pair].set(relaxed)

    # 3. Post-integration: recover rates from motor displacement:
    motor, rate = post_integrate(chain.motor, motor, dt)

    # 4. Resolve velocity constraints:
    for joints in partitions:
        pair = joints.bodies
        resolved = project_velocity_constraint(motor[pair], rate[pair], joints.anchors, chain.inverse_inertia[pair])
        rate = rate.at[pair].set(resolved)

    return chain._replace(motor=motor, rate=rate)


def advance(chain: Chain, partitions: list[Joints], n_substeps: int, dt: float) -> Chain:
    """The chain dt later, in equal substeps."""
    for _ in range(n_substeps):
        chain = step(chain, partitions, dt / n_substeps)
    return chain


def joint_gaps(motor: Motor, partitions: list[Joints]) -> Scalar:
    """The Euclidean separation of the two anchors of every joint: the norm of their join."""
    def gap(joints: Joints) -> Scalar:
        world_anchors = motor[joints.bodies] >> joints.anchors
        # The join of two points is a simple line: its norm has no pseudoscalar part.
        return (world_anchors[0] & world_anchors[1]).norm()

    return concatenate([gap(joints) for joints in partitions])
