"""Extended position-based dynamics of a chain of rigid links in 3D projective geometric algebra.

Each step integrates the links freely under gravity and Rayleigh damping, then closes every joint that has drifted apart. A joint constrains two body-fixed anchor points to coincide. Joining the two world anchors forms a line whose norm measures the separation and whose normalized direction is the unit forque along which the correcting impulse acts. Pulled into each body's frame, that unit forque maps through the inverse inertia into a body twist, and pairing the twist with the forque yields the body's compliance along the constraint line. One scalar impulse multiplier closes the gap across the summed compliances, and each body updates its pose by the exponential of its impulsive twist. After Verlet rate recovery from the pose change, a second impulse projection along the anchors' relative velocity line cancels relative anchor motion. A pinned body carries zero inverse inertia, so no impulse ever moves it.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import concatenate
from numga.algebras import PGA3D
from examples.mechanics import lie_integrators as lie

ga = PGA3D
Scalar = ga.gatype.scalar()                                                     # Scalar+
Point = ga.gatype.antivector()                                                  # Antivector-
Motor = ga.gatype.rotor()                                                       # Rotor+
Rate = ga.gatype.bivector()                                                     # Bivector+
Forque = ga.gatype.antibivector()                                               # Antibivector-
Inertia = ga.gatype((Forque, Rate))                                             # Forque- <- Rate+
InverseInertia = ga.gatype((Rate, Forque))                                      # Rate+ <- Forque-

PairPartition = tuple[np.ndarray, Point]


# --- math -----------------------------------------------------------------------------
def position_projection(
    motors: Motor, anchors: Point, inverse_inertias: InverseInertia, compliance: float, dt: float,
) -> Motor:
    """Distance-constraint projection moves each body pair (`[2, ...] Rotor+`) along the line between its anchors."""
    world_anchors = motors >> anchors                                           # [2, ...] Point-
    # Joining the two world anchors gives the separation as its norm and the impulse line as its normalized direction;
    # the tiny offset keeps closed joints at zero impulse without branching.
    line = world_anchors[0] & world_anchors[1]                                  # [...] Forque-
    separation = line.norm()                                                    # [...] Scalar+
    unit_forque = motors << (line / (separation + 1e-24))                       # [2, ...] Forque-
    # Mapping the unit forque through each body's inverse inertia yields its twist response, and pairing back gives the
    # effective inverse mass along the line.
    twists = inverse_inertias(unit_forque)                                      # [2, ...] Rate+
    total_compliance = (twists & unit_forque).sum(axis=0) + compliance / dt**2 + 1e-24   # [...] Scalar+
    impulse = twists * (separation / total_compliance * np.array([[1], [-1]]))  # [2, ...] Rate+
    return motors * (impulse * -0.5).exp()


def velocity_projection(
    motors: Motor, rates: Rate, anchors: Point, inverse_inertias: InverseInertia,
) -> Rate:
    """Velocity-constraint projection cancels relative anchor motion across each body pair (`[2, ...] Rate+`)."""
    # Joining each anchor with its body-frame velocity produces its world velocity line when pushed forward by the motor.
    anchor_velocities = motors >> anchors.regressive(anchors.commutator(rates))   # [2, ...] Forque-
    relative_velocity = anchor_velocities[1] - anchor_velocities[0]             # [...] Forque-
    speed = relative_velocity.scalar_norm_squared().square_root()               # [...] Scalar+
    unit_forque = motors << (relative_velocity / (speed + 1e-24))               # [2, ...] Forque-
    twists = inverse_inertias(unit_forque)                                      # [2, ...] Rate+
    total_compliance = (twists & unit_forque).sum(axis=0) + 1e-24               # [...] Scalar+
    impulse = twists * (speed / total_compliance * np.array([[1], [-1]]))       # [2, ...] Rate+
    return rates + impulse


def substep(
    motors: Motor,
    rates: Rate,
    first_moments: Point,
    inertias: Inertia,
    inverse_inertias: InverseInertia,
    gravity: Point,
    compliance: float,
    damping: float,
    partitions: list[PairPartition],
    dt: float,
) -> tuple[Motor, Rate]:
    """One XPBD substep advances the body poses (`[...] Rotor+`) and body-frame rates (`[...] Rate+`) by `dt`."""
    def external_forque(motor: Motor, rate: Rate) -> Forque:
        # Gravity acts along the line joining the mass-weighted centre to the body-frame gravity direction, while
        # Rayleigh damping opposes momentum proportionally.
        weight = first_moments & (motor << gravity)                             # [...] Forque-
        return weight - inertias(rate) * damping

    relaxed_motors, _ = lie.explicit_rk4(motors, rates, inertias, inverse_inertias, dt, external_forque)
    for pairs, anchors in partitions:
        projected = position_projection(
            relaxed_motors[pairs], anchors, inverse_inertias[pairs], compliance, dt,
        )
        relaxed_motors = relaxed_motors.at[pairs].set(projected)

    # Recovering body rates from the finite motor displacement keeps velocities consistent with the projected poses.
    next_motors = relaxed_motors.normalized().retyped(Motor)                    # [...] Rotor+
    next_rates = ((~motors * next_motors).log() * (-2 / dt)).retyped(Rate)      # [...] Rate+
    for pairs, anchors in partitions:
        resolved = velocity_projection(
            next_motors[pairs], next_rates[pairs], anchors, inverse_inertias[pairs],
        )
        next_rates = next_rates.at[pairs].set(resolved).retyped(Rate)
    return next_motors, next_rates


def advance(
    motors: Motor,
    rates: Rate,
    first_moments: Point,
    inertias: Inertia,
    inverse_inertias: InverseInertia,
    gravity: Point,
    compliance: float,
    damping: float,
    partitions: list[PairPartition],
    n_substeps: int,
    dt: float,
) -> tuple[Motor, Rate]:
    """`n_substeps` equal XPBD substeps advance the body poses and rates across one frame interval `dt`."""
    for _ in range(n_substeps):
        motors, rates = substep(
            motors, rates, first_moments, inertias, inverse_inertias,
            gravity, compliance, damping, partitions, dt / n_substeps,
        )
    return motors, rates


def swing_motors(
    motors: Motor,
    rates: Rate,
    first_moments: Point,
    inertias: Inertia,
    inverse_inertias: InverseInertia,
    gravity: Point,
    compliance: float,
    damping: float,
    partitions: list[PairPartition],
    n_steps: int,
    n_substeps: int,
    dt: float,
) -> Iterator[Motor]:
    """Yields the body poses (`[n_bodies] Rotor+`) initially and after each of `n_steps` frame intervals."""
    yield motors
    for _ in range(n_steps):
        motors, rates = advance(
            motors, rates, first_moments, inertias, inverse_inertias,
            gravity, compliance, damping, partitions, n_substeps, dt,
        )
        yield motors


def joint_separations(motors: Motor, partitions: list[PairPartition]) -> Scalar:
    """Euclidean anchor separations (`[...] Scalar+`) across all partitions come from the norm of the joined anchors."""
    def partition_separations(partition: PairPartition) -> Scalar:
        pairs, anchors = partition
        world_anchors = motors[..., pairs] >> anchors                           # [..., 2, ...] Point-
        return (world_anchors[..., 0, :] & world_anchors[..., 1, :]).norm()

    return concatenate([partition_separations(partition) for partition in partitions], axis=-1)
