"""Lie-group integrators for rigid bodies, on motors and rates.

The state of a body is a motor in the Lie group, its pose, and a rate in the Lie algebra,
a bivector of angular and linear velocity in the body frame. Inertia maps a rate to its
momentum, a forque; its inverse maps a forque back to a rate. External forques come from a
function of the motor and the rate.

Methods implemented:
- Explicit symplectic Verlet (Lie-Euler)
- Explicit Runge-Kutta 4 on Lie groups
- Explicit 4th-order Munthe-Kaas (RKMK4) with polynomial dexpinv
- Explicit Lie-Newmark method
- Variational Lie-Verlet method

The last two solve an implicit step by Newton's method, differentiated by JAX.

The algebra is not fixed here: each function reads it from its arguments, so the same lines
integrate rotors in any dimension as well as motors of PGA.

References:
- Hairer, Lubich, Wanner: Geometric Numerical Integration
- Krysl: Explicit Newmark-type integrators on Lie groups
"""

from __future__ import annotations

from typing import Any, Callable

from numga import Extensor


# --- plumbing -------------------------------------------------------------------------
def RK4(f: Callable, y: Any, h: float) -> Any:
    """Classical 4th-order Runge-Kutta step."""
    k1 = f(y)
    k2 = f(y + 0.5 * h * k1)
    k3 = f(y + 0.5 * h * k2)
    k4 = f(y + h * k3)
    return y + (h / 3.0) * (k2 + k3 + (k1 + k4) * 0.5)


def newton(func: Callable, init: Extensor, iterations: int = 10) -> Extensor:
    """Solve `func(x) == 0` from init by Newton steps, with the Jacobian of the kernel from JAX.

    Only the implicit integrators call this, so only they need JAX installed.
    """
    import jax
    import jax.numpy as jnp

    def residual(x):
        return func(init.context.extensor(init.gatype, x)).kernel

    jacobian = jax.jacfwd(residual)

    def step(i, x):
        return x - jnp.linalg.solve(jacobian(x), residual(x))

    solved = jax.lax.fori_loop(0, iterations, step, init.kernel)
    return init.context.extensor(init.gatype, solved)


# --- math -----------------------------------------------------------------------------
def inertia_from_points(points: Extensor) -> tuple[Extensor, Extensor]:
    """Construct inertia and its inverse from a point cloud of unit masses.

    Each point contributes the rate-to-momentum extensor:
        points & points.commutator(points.algebra.gatype.bivector())
    the regressive product of the point with its commutator with an open bivector.
    """
    inertia = (points & points.commutator(points.algebra.gatype.bivector())).sum(axis=-1)
    return inertia, inertia.inverse()


def kinetic_energy(rate: Extensor, inertia: Extensor) -> Extensor:
    """Compute rigid body kinetic energy: 0.5 * (rate & inertia(rate))."""
    momentum = inertia(rate)
    return (momentum & rate) * 0.5


def free(motor: Extensor, rate: Extensor) -> Extensor:
    """No external forque: a free body."""
    return rate.dual() * 0.0


def _net_forque(forque: Callable, inertia: Extensor, motor: Extensor, rate: Extensor) -> Extensor:
    """External forque minus the gyroscopic term: `forque(motor, rate) - inertia(rate).commutator(rate)`."""
    gyro = inertia(rate).commutator(rate)
    return forque(motor, rate).cast(gyro.gatype.output_subspace) - gyro


def explicit_verlet(
    motor: Extensor, rate: Extensor, inertia: Extensor, inertia_inv: Extensor, dt: float, forque: Callable,
) -> tuple[Extensor, Extensor]:
    """Verlet pre- and post-integration step (unconstrained).

    Steps rate via RK4, steps motor along the twist, and recovers the rate
    from the relative motor displacement.
    """
    def dr(r: Extensor) -> Extensor:
        return inertia_inv(_net_forque(forque, inertia, motor, r))

    rate_pred = RK4(dr, rate, dt)
    motor_next = (motor * (rate_pred * (-dt / 2.0)).exp()).normalized()
    rate_recovered = (~motor * motor_next).log() * (-2.0 / dt)
    return motor_next, rate_recovered


def explicit_rk4(
    motor: Extensor, rate: Extensor, inertia: Extensor, inertia_inv: Extensor, dt: float, forque: Callable,
) -> tuple[Extensor, Extensor]:
    """Explicit RK4 on the rate, then a motor step with the stepped rate."""
    def dr(r: Extensor) -> Extensor:
        return inertia_inv(_net_forque(forque, inertia, motor, r))

    rate_next = RK4(dr, rate, dt)
    motor_next = motor * (rate_next * (-dt / 2.0)).exp()
    return motor_next, rate_next


def explicit_rkmk4(
    motor: Extensor, rate: Extensor, inertia: Extensor, inertia_inv: Extensor, dt: float, forque: Callable,
) -> tuple[Extensor, Extensor]:
    """Explicit 4th-order Munthe-Kaas integration of Lie state.

    The motor step is `motor * h.exp()`, with the bivector h integrated in the Lie algebra by
    classical RK4. The vector field for h is the body rate corrected by dexpinv, a polynomial in
    the adjoint `h.commutator(bivector) * 2`, the commutator with an open bivector slot. Nothing
    here depends on the dimension: the same lines integrate rotors in any dimension.
    """
    def dr(m: Extensor, r: Extensor) -> Extensor:
        return inertia_inv(_net_forque(forque, inertia, m, r))

    def dh(h: Extensor, r: Extensor) -> Extensor:
        # The time derivative of `motor * h.exp()` must equal that motor times `r * -0.5`;
        # inverting the derivative of the exponential gives the rate of h, `dexpinv(r * -0.5)`.
        bivector = h.algebra.subspace.bivector()
        ad = h.commutator(bivector) * 2.0
        dexpinv = bivector + ad * 0.5 + ad(ad) * (1.0 / 12.0)
        return dexpinv(r * -0.5)

    # The state (h, rate) is a pair of bivectors: stack them and let the plain RK4 step it.
    def derivative(state: Extensor) -> Extensor:
        h, r = state[0], state[1]
        return Extensor.stack([dh(h, r), dr(motor * h.exp(), r)])

    state = RK4(derivative, Extensor.stack([rate * 0.0, rate]), dt)
    h, rate_next = state[0], state[1]
    motor_next = motor * h.exp()
    return motor_next, rate_next


def explicit_lie_newmark(
    motor: Extensor, rate: Extensor, inertia: Extensor, inertia_inv: Extensor, dt: float, forque: Callable,
) -> tuple[Extensor, Extensor]:
    """Explicit Lie-Newmark symplectic integrator."""
    def impulse(m: Extensor, r: Extensor) -> Extensor:
        return _net_forque(forque, inertia, m, r) * (dt / 2.0)

    half_rate_step = inertia_inv(impulse(motor, rate))
    rate_half = rate + half_rate_step
    motor_new = (rate_half * (dt / 2.0)).exp() * motor

    def implicit(rn: Extensor) -> Extensor:
        return -rn + rate_half + inertia_inv(impulse(motor_new, rn))

    rate_new = newton(implicit, rate_half + half_rate_step)
    return motor_new, rate_new


def variational_lie_verlet(
    motor: Extensor, rate: Extensor, inertia: Extensor, inertia_inv: Extensor, dt: float, forque: Callable,
) -> tuple[Extensor, Extensor]:
    """Variational Lie-Verlet integrator."""
    def energy(r: Extensor) -> Extensor:
        p = inertia(r)
        return (p.wedge(r) * r).restrict[2]

    def net(m: Extensor, r: Extensor) -> Extensor:
        return _net_forque(forque, inertia, m, r)

    def implicit(rh: Extensor) -> Extensor:
        return -rh + rate + inertia_inv(net(motor, rh) - energy(rh)) * (dt / 2.0)

    rate_half = newton(implicit, rate)
    motor_new = motor * (rate_half * (-dt / 4.0)).exp()
    rate_new = rate_half + inertia_inv(net(motor_new, rate_half) + energy(rate_half)) * (dt / 2.0)
    return motor_new, rate_new
