"""Geometric Lie-group integrators and dynamics for rigid bodies in Geometric Algebra.

Operates directly on Extensor state without object-oriented wrappers:
- motor: rotor/motor in the Lie group describing orientation/pose
- rate: bivector in the Lie algebra describing angular/linear velocity
- inertia: physical inertia map (AntiBivector <- Bivector)
- inertia_inv: inverse inertia map (Bivector <- AntiBivector)

Methods implemented:
- Explicit symplectic Verlet (Lie-Euler)
- Explicit Runge-Kutta 1 and 4 on Lie groups
- Explicit 4th-order Munthe-Kaas (RKMK4) with polynomial dexpinv
- Explicit Lie-Newmark method
- Variational Lie-Verlet method
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


def inertia_from_points(points: Extensor, masses: Extensor | None = None) -> tuple[Extensor, Extensor]:
    """Construct inertia and its inverse from point cloud coordinates.

    Each point p contributes the rate-to-momentum extensor:
        p & (p x Bivector)
    where & is the regressive product and x is the commutator.
    """
    ga = points.algebra
    Bivector = ga.gatype.bivector()
    if masses is not None:
        inertia = (points & points.commutator(Bivector) * masses).sum(axis=-1)
    else:
        inertia = (points & points.commutator(Bivector)).sum(axis=-1)
    return inertia, inertia.inverse()


def kinetic_energy(rate: Extensor, inertia: Extensor) -> Extensor:
    """Compute rigid body kinetic energy: 0.5 * (rate & inertia(rate))."""
    momentum = inertia(rate)
    return (momentum & rate) * 0.5


def _zero_forque(motor: Extensor, rate: Extensor) -> Extensor:
    return 0.0 * rate


def _net_forque(
    ext_forque: Callable[[Extensor, Extensor], Extensor],
    inertia: Extensor,
    motor: Extensor,
    rate: Extensor,
) -> Extensor:
    """External forque minus the gyroscopic term: F_ext - (I(rate) x rate)."""
    gyro = inertia(rate).commutator(rate)
    return ext_forque(motor, rate).cast(gyro.gatype.output_subspace) - gyro


def rate_derivative(
    rate: Extensor,
    inertia: Extensor,
    inertia_inv: Extensor,
    forque: Extensor | None = None,
) -> Extensor:
    """Generalized Euler rotational equation in the Lie algebra:
        d/dt rate = I^-1(forque - I(rate) x rate)
    """
    gyro = inertia(rate).commutator(rate)
    net = (forque.cast(gyro.gatype.output_subspace) - gyro) if forque is not None else (-gyro)
    return inertia_inv(net)


def explicit_verlet(
    motor: Extensor,
    rate: Extensor,
    inertia: Extensor,
    inertia_inv: Extensor,
    dt: float,
    ext_forque: Callable[[Extensor, Extensor], Extensor] = _zero_forque,
) -> Tuple[Extensor, Extensor]:
    """Verlet pre- and post-integration step (unconstrained).

    Steps rate via RK4, steps motor along the twist, and recovers the rate
    from the relative motor displacement.
    """
    def dr(r: Extensor) -> Extensor:
        forque = _net_forque(ext_forque, inertia, motor, r)
        return inertia_inv(forque)

    rate_pred = RK4(dr, rate, dt)
    motor_next = (motor * (rate_pred * (-dt / 2.0)).exp()).normalized()
    rate_recovered = (~motor * motor_next).log() * (-2.0 / dt)
    return motor_next, rate_recovered


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

    The motor step is motor * exp(H), with H integrated in the Lie algebra by classical RK4.
    The vector field for H is the body rate corrected by dexpinv, a polynomial in the adjoint
    ad_H = [H, .], which is the commutator with an open bivector slot.
    """
    bivector = rate.context.algebra.subspace.bivector()

    def dr(m: Extensor, r: Extensor) -> Extensor:
        forque = _net_forque(ext_forque, inertia, m, r)
        return inertia_inv(forque)

    def dh(h: Extensor, r: Extensor) -> Extensor:
        ad = h.commutator(bivector) * 2.0
        dexpinv = bivector + ad * 0.5 + ad(ad) * (1.0 / 12.0)
        return dexpinv(r * -0.5)

    def derivative(state: Extensor) -> Extensor:
        h, r = state[0], state[1]
        return Extensor.stack([dh(h, r), dr(motor * h.exp(), r)])

    state = RK4(derivative, Extensor.stack([rate * 0.0, rate]), dt)
    h, rate_next = state[0], state[1]
    motor_next = motor * h.exp()
    return motor_next, rate_next


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
