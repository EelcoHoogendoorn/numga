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

Run from rewrite/ with:
    PYTHONPATH=src:. python -m examples.mechanics.xpbd
"""

from __future__ import annotations

from typing import Tuple
import numpy as np

from numga import Extensor
from examples.mechanics.integrators import RK4


def project_distance_constraint(
    motors: Extensor,
    anchors: Extensor,
    inertia_inv: Extensor,
    compliance: Extensor,
    dt: float,
) -> Extensor:
    """Project a pairwise distance / spherical joint constraint between two bodies.

    Parameters
    ----------
    motors : Extensor
        Shape `[2, ...]`, current body motors (rotors/translators).
    anchors : Extensor
        Shape `[2, ...]`, anchor coordinates in body-local frame (antivector points).
    inertia_inv : Extensor
        Shape `[2, ...]`, inverse inertia maps (Bivector <- AntiBivector).
    compliance : Extensor
        Shape `[...]`, constraint inverse stiffness alpha.
    dt : float
        Time step.

    Returns
    -------
    Extensor
        Shape `[2, ...]`, updated motors satisfying the constraint.
    """
    # 1. Transform anchor points to world frame via motor pushforward:
    world_anchors = motors >> anchors

    # 2. Join the two world points into a line:
    # In PGA, the regressive product (&) of two points is the line connecting them.
    # Its norm is the Euclidean distance between the anchors.
    line = world_anchors[0] & world_anchors[1]
    magnitude = line.norm()
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
    motors: Extensor,
    rates: Extensor,
    anchors: Extensor,
    inertia_inv: Extensor,
    dt: float,
) -> Extensor:
    """Resolve velocity constraint impulses at anchor points.

    Damps relative velocity between anchor points along the constraint line.
    """
    context = anchors.context
    bivector = context.algebra.subspace.bivector()

    # Relative anchor velocity in world coordinates:
    anchors_map = anchors & anchors.commutator(bivector)
    velocities = motors >> anchors_map(rates)
    forque = velocities[1] - velocities[0]
    magnitude = forque.norm()
    direction = forque / (magnitude + 1e-24)

    local_dir = motors << direction
    steps = inertia_inv(local_dir)
    inertial_compliances = steps & local_dir
    total_compliance = inertial_compliances.sum(axis=0) + 1e-24
    multiplier = (magnitude / total_compliance) * np.array([[+1], [-1]])
    impulse = steps * multiplier
    return rates + impulse


def external_forque(
    motor: Extensor,
    rate: Extensor,
    first_moment: Extensor,
    gravity: Extensor,
    damping: Extensor,
) -> Extensor:
    """Compute external forque line in body-local frame (gravity + damping)."""
    gravity_local = motor << gravity
    grav_forque = first_moment & gravity_local
    damp_forque = -(rate * damping).dual()
    return grav_forque + damp_forque


def pre_integrate(
    motor: Extensor,
    rate: Extensor,
    inertia: Extensor,
    inertia_inv: Extensor,
    first_moment: Extensor,
    gravity: Extensor,
    damping: Extensor,
    dt: float,
) -> Tuple[Extensor, Extensor]:
    """Verlet pre-integration: unconstrained inertial state update."""
    def dr(r: Extensor) -> Extensor:
        forque = external_forque(motor, r, first_moment, gravity, damping)
        gyro = inertia(r).commutator(r)
        return inertia_inv(forque.cast(gyro.gatype.output_subspace) - gyro)

    rate_pred = RK4(dr, rate, dt)
    motor_pred = motor * (rate_pred * (-dt / 2.0)).exp()
    return motor_pred, rate_pred


def post_integrate(
    old_motor: Extensor,
    relaxed_motor: Extensor,
    dt: float,
) -> Tuple[Extensor, Extensor]:
    """Verlet post-integration: update rates from relaxed motors."""
    motor_normalized = relaxed_motor.normalized()
    rate = (~old_motor * motor_normalized).log() * (-2.0 / dt)
    return motor_normalized, rate


def main(
    n_bodies: int = 6,
    n_steps: int = 60,
    substeps: int = 5,
    dt: float = 0.02,
    plot_path: str | None = None,
) -> Tuple[list, np.ndarray]:
    """Simulate a swinging chain using XPBD and check constraint satisfaction."""
    import matplotlib.pyplot as plt
    from examples import PLOT_DIR
    from examples.mechanics.xpbd_plumbing import setup_chain, simulate_chain, draw_chain

    if plot_path is None:
        plot_path = str(PLOT_DIR / "xpbd_chain.png")

    state, chain_constraints = setup_chain(n_bodies=n_bodies)
    states, violations = simulate_chain(
        state,
        chain_constraints,
        n_steps=n_steps,
        substeps=substeps,
        dt=dt,
    )

    # Check constraint satisfaction: residual anchor separation should be negligible
    max_violation = float(np.max(violations[-1]))
    print(f"Chain of {n_bodies} bodies: final max joint violation = {max_violation:.2e} m")
    assert max_violation < 1e-4, f"Constraint violation too large: {max_violation}"

    fig = draw_chain(states, chain_constraints, plot_path)
    return states, violations


if __name__ == "__main__":
    main()
