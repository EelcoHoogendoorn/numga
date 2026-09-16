"""Geometric Lie-group integrators for rigid body dynamics.

Implements variational and explicit Lie-group numerical integrators:
- Explicit RK1 and RK4 on Lie groups
- Explicit 4th-order Munthe-Kaas (RKMK4), with dexpinv as a polynomial in the adjoint
- Explicit Lie-Newmark method
- Variational Lie-Verlet method
- Newton-Raphson solver wrapper for implicit steps using JAX autodiff

References:
- Hairer, Lubich, Wanner: Geometric Numerical Integration
- Krysl: Explicit Newmark-type integrators on Lie groups
"""

from __future__ import annotations

from typing import Callable, Tuple
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

from numga import Extensor
from examples.mechanics.integrators import RK1, RK4



def newton_solver(fn: Callable, n: int = 10) -> Callable:
    """Return a callable that performs n Newton steps on a vector function using JAX."""
    if not HAS_JAX:
        raise RuntimeError("newton_solver requires JAX to be installed")
    jac_fn = jax.jacfwd(fn)

    def step(i, x):
        return x - jnp.linalg.solve(jac_fn(x), fn(x))

    return lambda x: jax.lax.fori_loop(0, n, step, x)


def newton_solver_wrap(func: Callable[[Extensor], Extensor], init: Extensor, n: int = 10) -> Extensor:
    """Wrap Newton solve of an Extensor function."""
    def func_wrap(x):
        x_ext = init.context.extensor(init.gatype, x)
        return func(x_ext).kernel

    solver = newton_solver(func_wrap, n=n)
    solved_kernel = solver(init.kernel)
    return init.context.extensor(init.gatype, solved_kernel)


def _zero_forque(motor: Extensor, rate: Extensor) -> Extensor:
    return 0.0 * rate


def _net_forque(
    ext_forque: Callable[[Extensor, Extensor], Extensor],
    inertia: Extensor,
    motor: Extensor,
    rate: Extensor,
) -> Extensor:
    """External forque minus the gyroscopic term, in the momentum's own layout."""
    gyro = inertia(rate).commutator(rate)
    return ext_forque(motor, rate).cast(gyro.gatype.output_subspace) - gyro


def explicit_rk1(
    motor: Extensor,
    rate: Extensor,
    inertia: Extensor,
    inertia_inv: Extensor,
    dt: float,
    ext_forque: Callable[[Extensor, Extensor], Extensor] = _zero_forque,
) -> Tuple[Extensor, Extensor]:
    """Explicit RK1 (Euler) integration of Lie state."""
    def dr(r: Extensor) -> Extensor:
        forque = _net_forque(ext_forque, inertia, motor, r)
        return inertia_inv(forque)

    rate_next = RK1(dr, rate, dt)
    motor_next = motor * (rate_next * (-dt / 2.0)).exp()
    return motor_next, rate_next


def explicit_rk4(
    motor: Extensor,
    rate: Extensor,
    inertia: Extensor,
    inertia_inv: Extensor,
    dt: float,
    ext_forque: Callable[[Extensor, Extensor], Extensor] = _zero_forque,
) -> Tuple[Extensor, Extensor]:
    """Explicit RK4 integration of Lie state."""
    def dr(r: Extensor) -> Extensor:
        forque = _net_forque(ext_forque, inertia, motor, r)
        return inertia_inv(forque)

    rate_next = RK4(dr, rate, dt)
    motor_next = motor * (rate_next * (-dt / 2.0)).exp()
    return motor_next, rate_next


def explicit_rkmk4(
    motor: Extensor,
    rate: Extensor,
    inertia: Extensor,
    inertia_inv: Extensor,
    dt: float,
    ext_forque: Callable[[Extensor, Extensor], Extensor] = _zero_forque,
) -> Tuple[Extensor, Extensor]:
    """Explicit 4th-order Munthe-Kaas integration of Lie state.

    The motor step is motor * exp(H), matching Body.motor_add_step, with H integrated in
    the Lie algebra by classical RK4. The vector field for H is the body rate corrected by
    dexpinv, a polynomial in the adjoint ad_H = [H, ·], which is the commutator with an open
    bivector slot. Nothing here depends on the dimension: the same lines integrate rotors
    of Spin(n) for any n.
    """
    bivector = rate.context.algebra.subspace.bivector()

    def dr(m: Extensor, r: Extensor) -> Extensor:
        forque = _net_forque(ext_forque, inertia, m, r)
        return inertia_inv(forque)

    def dh(h: Extensor, r: Extensor) -> Extensor:
        # d/dt (M0 exp H) = M0 exp(H) dexp_{-H}(H') must equal M (-r/2), so H' = dexpinv_{-H}(-r/2)
        ad = h.commutator(bivector) * 2.0
        dexpinv = bivector + ad * 0.5 + ad(ad) * (1.0 / 12.0)
        return dexpinv(r * -0.5)

    # The state (H, rate) is a pair of bivectors: stack them and let the plain RK4 step it.
    def derivative(state: Extensor) -> Extensor:
        h, r = state[0], state[1]
        return Extensor.stack([dh(h, r), dr(motor * h.exp(), r)])

    state = RK4(derivative, Extensor.stack([rate * 0.0, rate]), dt)
    h, rate_next = state[0], state[1]
    motor_next = motor * h.exp()
    return motor_next, rate_next


def explicit_lie_newmark(
    motor: Extensor,
    rate: Extensor,
    inertia: Extensor,
    inertia_inv: Extensor,
    dt: float,
    ext_forque: Callable[[Extensor, Extensor], Extensor] = _zero_forque,
) -> Tuple[Extensor, Extensor]:
    """Explicit Lie-Newmark symplectic integrator."""
    def impulse(m: Extensor, r: Extensor) -> Extensor:
        forque = _net_forque(ext_forque, inertia, m, r)
        return forque * (dt / 2.0)

    half_rate_step = inertia_inv(impulse(motor, rate))
    rate_half = rate + half_rate_step
    motor_new = (rate_half * (dt / 2.0)).exp() * motor

    def implicit(rn: Extensor) -> Extensor:
        return -rn + rate_half + inertia_inv(impulse(motor_new, rn))

    rate_new = newton_solver_wrap(implicit, rate_half + half_rate_step)
    return motor_new, rate_new


def variational_lie_verlet(
    motor: Extensor,
    rate: Extensor,
    inertia: Extensor,
    inertia_inv: Extensor,
    dt: float,
    ext_forque: Callable[[Extensor, Extensor], Extensor] = _zero_forque,
) -> Tuple[Extensor, Extensor]:
    """Variational Lie-Verlet integrator."""
    def energy(r: Extensor) -> Extensor:
        p = inertia(r)
        return (p.wedge(r) * r).restrict[2]

    def forque(m: Extensor, r: Extensor) -> Extensor:
        return _net_forque(ext_forque, inertia, m, r)

    def implicit(rh: Extensor) -> Extensor:
        return -rh + rate + inertia_inv(forque(motor, rh) - energy(rh)) * (dt / 2.0)

    rate_half = newton_solver_wrap(implicit, init=rate)
    motor_new = motor * (rate_half * (-dt / 4.0)).exp()
    rate_new = rate_half + inertia_inv(forque(motor_new, rate_half) + energy(rate_half)) * (dt / 2.0)
    return motor_new, rate_new
