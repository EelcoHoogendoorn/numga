"""Lie-group integrators step rigid-body poses and body-frame rates across any geometric algebra.

A rigid body's pose is a rotor or motor (`Rotor+`) in the even subalgebra, and its body-frame rate is a bivector (`Bivector+`) combining angular and linear velocity. The body's inertia map sends rates to momenta (`Antibivector- <- Bivector+`), its inverse maps net forques back to rate accelerations, and external forques depend on the current motor and rate.

Five stepping schemes share one signature:
- Explicit symplectic Verlet (`explicit_verlet`)
- Explicit fourth-order Runge-Kutta on the rate (`explicit_rk4`)
- Explicit fourth-order Munthe-Kaas with truncated `dexpinv` (`explicit_rkmk4`)
- Implicit-rate symplectic Lie-Newmark (`explicit_lie_newmark`)
- Implicit-midpoint variational Lie-Verlet (`variational_lie_verlet`)

The last two solve their implicit rate equations with fixed-count Newton iterations differentiated by JAX. Every function reads the algebra from its extensor inputs, so the same code integrates Euclidean rotors in any dimension as well as projective motors, following Hairer, Lubich, and Wanner (*Geometric Numerical Integration*) and Krysl (*Explicit Newmark-type integrators on Lie groups*).
"""

from __future__ import annotations

from typing import Any, Callable

from numga import Extensor, stack

# Fixed Newton iteration count for each implicit rate solve.
N_NEWTON_STEPS = 10


# --- plumbing -------------------------------------------------------------------------
def RK4(f: Callable, y: Any, h: float) -> Any:
    """Classical fourth-order Runge-Kutta advances `y` across step `h` under vector field `f`."""
    stage_1 = f(y)
    stage_2 = f(y + 0.5 * h * stage_1)
    stage_3 = f(y + 0.5 * h * stage_2)
    stage_4 = f(y + h * stage_3)
    return y + (h / 3) * (stage_2 + stage_3 + (stage_1 + stage_4) * 0.5)


def newton(func: Callable, init: Extensor) -> Extensor:
    """Fixed-count Newton iterations find a root of `func(x) == 0` starting from `init` using a JAX forward Jacobian.

    Only the implicit integrators call `newton`, so explicit integrators run without JAX.
    """
    import jax
    import jax.numpy as jnp

    def residual(kernel):
        return func(init.context.extensor(init.gatype, kernel)).kernel

    jacobian = jax.jacfwd(residual)

    def newton_step(step_index, kernel):
        return kernel - jnp.linalg.solve(jacobian(kernel), residual(kernel))

    root_kernel = jax.lax.fori_loop(0, N_NEWTON_STEPS, newton_step, init.kernel)
    return init.context.extensor(init.gatype, root_kernel)                           # [] Bivector+


# --- math -----------------------------------------------------------------------------
def inertia_from_points(points: Extensor) -> tuple[Extensor, Extensor]:
    """Unit-mass points (`[..., n_points] Point-`) sum to a body's inertia map (`Antibivector- <- Bivector+`) and its inverse."""
    Rate = points.algebra.gatype.bivector()
    inertia = points.regressive(points.commutator(Rate)).sum(axis=-1)                # [...] Antibivector- <- Bivector+
    return inertia, inertia.inverse()


def kinetic_energy(rate: Extensor, inertia: Extensor) -> Extensor:
    """Pairing the body's momentum with its rate gives twice the kinetic energy (`Scalar+`)."""
    momentum = inertia(rate)                                                         # [...] Antibivector-
    return (rate & momentum) * 0.5                                                   # [...] Scalar+


def free(motor: Extensor, rate: Extensor) -> Extensor:
    """Zero external forque (`Antibivector-`) models torque-free and force-free motion."""
    return rate.dual() * 0                                                           # [...] Antibivector-


def _net_forque(forque: Callable, inertia: Extensor, motor: Extensor, rate: Extensor) -> Extensor:
    """Subtracting the Euler gyroscopic commutator from the external forque gives the net body-frame forque (`Antibivector-`)."""
    gyroscopic_forque = inertia(rate).commutator(rate)                               # [...] Antibivector-
    return forque(motor, rate) - gyroscopic_forque                                   # [...] Antibivector-


def explicit_verlet(
    motor: Extensor, rate: Extensor, inertia: Extensor, inverse_inertia: Extensor, dt: float, forque: Callable,
) -> tuple[Extensor, Extensor]:
    """An RK4 rate prediction steps the pose motor and recovers the updated rate from the relative motor logarithm."""
    def rate_acceleration(candidate_rate: Extensor) -> Extensor:
        return inverse_inertia(_net_forque(forque, inertia, motor, candidate_rate))  # [...] Bivector+

    predicted_rate = RK4(rate_acceleration, rate, dt)                                # [...] Bivector+
    next_motor = (motor * (predicted_rate * (-dt / 2)).exp()).normalized()           # [...] Rotor+
    recovered_rate = (~motor * next_motor).log() * (-2 / dt)                         # [...] Bivector+
    return next_motor, recovered_rate


def explicit_rk4(
    motor: Extensor, rate: Extensor, inertia: Extensor, inverse_inertia: Extensor, dt: float, forque: Callable,
) -> tuple[Extensor, Extensor]:
    """Explicit fourth-order Runge-Kutta updates the body rate before stepping the motor by exponential map."""
    def rate_acceleration(candidate_rate: Extensor) -> Extensor:
        return inverse_inertia(_net_forque(forque, inertia, motor, candidate_rate))  # [...] Bivector+

    next_rate = RK4(rate_acceleration, rate, dt)                                     # [...] Bivector+
    next_motor = motor * (next_rate * (-dt / 2)).exp()                               # [...] Rotor+
    return next_motor, next_rate


def explicit_rkmk4(
    motor: Extensor, rate: Extensor, inertia: Extensor, inverse_inertia: Extensor, dt: float, forque: Callable,
) -> tuple[Extensor, Extensor]:
    """Fourth-order Munthe-Kaas integrates the local bivector generator and body rate together in the Lie algebra.

    Stepping `motor * step_bivector.exp()` pulls the kinematic velocity back to the Lie algebra through the fourth-order `dexpinv` polynomial in the commutator map `step_bivector.commutator(Rate) * 2`.
    """
    def rate_acceleration(pose_motor: Extensor, candidate_rate: Extensor) -> Extensor:
        return inverse_inertia(_net_forque(forque, inertia, pose_motor, candidate_rate))  # [...] Bivector+

    def generator_rate(step_bivector: Extensor, candidate_rate: Extensor) -> Extensor:
        # Inverting the derivative of `exp` maps `candidate_rate * -0.5` to the Lie-algebra rate of `step_bivector`.
        Rate = step_bivector.algebra.gatype.bivector()
        adjoint_map = step_bivector.commutator(Rate) * 2                             # [...] Bivector+ <- Bivector+
        dexp_inverse = Rate + adjoint_map * 0.5 + adjoint_map(adjoint_map) * (1 / 12)  # [...] Bivector+ <- Bivector+
        return dexp_inverse(candidate_rate * -0.5)                                   # [...] Bivector+

    # Stacking `(step_bivector, rate)` lets classical RK4 advance both bivectors together.
    def state_rate(stacked_pair: Extensor) -> Extensor:
        step_bivector, candidate_rate = stacked_pair[0], stacked_pair[1]             # [...] Bivector+, Bivector+
        stage_motor = motor * step_bivector.exp()                                    # [...] Rotor+
        return stack([generator_rate(step_bivector, candidate_rate), rate_acceleration(stage_motor, candidate_rate)])  # [2, ...] Bivector+

    next_pair = RK4(state_rate, stack([rate * 0, rate]), dt)                         # [2, ...] Bivector+
    step_bivector, next_rate = next_pair[0], next_pair[1]                            # [...] Bivector+, Bivector+
    next_motor = motor * step_bivector.exp()                                         # [...] Rotor+
    return next_motor, next_rate


def explicit_lie_newmark(
    motor: Extensor, rate: Extensor, inertia: Extensor, inverse_inertia: Extensor, dt: float, forque: Callable,
) -> tuple[Extensor, Extensor]:
    """Symplectic Lie-Newmark advances the motor explicitly from the half-step rate and solves implicitly for the end-step rate."""
    def half_impulse(pose_motor: Extensor, candidate_rate: Extensor) -> Extensor:
        return _net_forque(forque, inertia, pose_motor, candidate_rate) * (dt / 2)   # [...] Antibivector-

    half_rate_increment = inverse_inertia(half_impulse(motor, rate))                 # [...] Bivector+
    half_rate = rate + half_rate_increment                                           # [...] Bivector+
    next_motor = (half_rate * (dt / 2)).exp() * motor                                # [...] Rotor+

    def rate_residual(candidate_rate: Extensor) -> Extensor:
        return -candidate_rate + half_rate + inverse_inertia(half_impulse(next_motor, candidate_rate))  # [...] Bivector+

    next_rate = newton(rate_residual, half_rate + half_rate_increment)               # [...] Bivector+
    return next_motor, next_rate


def variational_lie_verlet(
    motor: Extensor, rate: Extensor, inertia: Extensor, inverse_inertia: Extensor, dt: float, forque: Callable,
) -> tuple[Extensor, Extensor]:
    """Variational Lie-Verlet solves implicitly for the half-step rate before updating the pose motor and end-step rate."""
    def cayley_correction(candidate_rate: Extensor) -> Extensor:
        momentum = inertia(candidate_rate)                                           # [...] Antibivector-
        return (momentum.wedge(candidate_rate) * candidate_rate).restrict[2]         # [...] Antibivector-

    def net_forque(pose_motor: Extensor, candidate_rate: Extensor) -> Extensor:
        return _net_forque(forque, inertia, pose_motor, candidate_rate)              # [...] Antibivector-

    def half_rate_residual(candidate_half_rate: Extensor) -> Extensor:
        total_forque = net_forque(motor, candidate_half_rate) - cayley_correction(candidate_half_rate)  # [...] Antibivector-
        return -candidate_half_rate + rate + inverse_inertia(total_forque) * (dt / 2)  # [...] Bivector+

    half_rate = newton(half_rate_residual, rate)                                     # [...] Bivector+
    next_motor = motor * (half_rate * (-dt / 4)).exp()                               # [...] Rotor+
    end_forque = net_forque(next_motor, half_rate) + cayley_correction(half_rate)    # [...] Antibivector-
    next_rate = half_rate + inverse_inertia(end_forque) * (dt / 2)                   # [...] Bivector+
    return next_motor, next_rate
