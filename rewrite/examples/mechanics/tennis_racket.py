"""Test the tennis racket theorem, or medial axis theorem.

Demonstrates the Dzhanibekov effect / intermediate axis theorem:
A rigid body with 3 distinct principal moments of inertia rotates stably
around its major and minor axes, but tumbles unstably around its intermediate
(medial) axis.
"""

from __future__ import annotations

import numpy as np

from typing import Callable, NamedTuple

from numga.algebra import Algebra
from numga import Extensor, NumpyContext
from examples.mechanics.lie_integrators import (
    explicit_verlet,
    explicit_rk4,
    explicit_rkmk4,
    inertia_from_points,
)
from examples import PLOT_DIR


class RigidBodyState(NamedTuple):
    """Rigid body dynamical state in Geometric Algebra."""
    motor: Extensor
    rate: Extensor
    inertia: Extensor
    inertia_inv: Extensor


def make_n_cube(n: int) -> np.ndarray:
    """Vertices of an n-dimensional hypercube centered at origin."""
    bits = ((np.arange(2**n)[:, None] & (1 << np.arange(n))) > 0)
    return 2.0 * bits - 1.0


def make_n_rect(n: int) -> np.ndarray:
    """Asymmetric rectangular cuboid with distinct axis lengths."""
    return make_n_cube(n) * (np.arange(n) + 1.0)


def setup_tennis_racket_bodies(context: NumpyContext, seed: int) -> RigidBodyState:
    """Set up batched bodies spinning in each independent bivector plane.

    Works for p=2, 3, 4, 5.
    p > 3 is fascinating: some medial axes become seemingly chaotic,
    while stranger still, some medial axes actually stabilize.
    """
    ndim = context.algebra.dimension
    coords = make_n_rect(ndim)
    points = context.multivector.vector(coords).dual()

    nb = len(context.algebra.subspace.bivector())
    points_batched = points[None, :].broadcast_to((nb, len(coords)))
    inertia, inertia_inv = inertia_from_points(points_batched)
    motor = context.multivector.rotor().broadcast_to(nb)

    # Initial spin in each principal plane with a small perturbation
    rng = np.random.default_rng(seed)
    rates = np.eye(nb) + rng.normal(scale=1e-5, size=(nb, nb))
    rate = context.multivector.bivector(rates)
    return RigidBodyState(motor, rate, inertia, inertia_inv)


def verlet_step(body: RigidBodyState, dt: float) -> RigidBodyState:
    """Explicit Lie-Verlet / symplectic Euler step."""
    motor, rate = explicit_verlet(body.motor, body.rate, body.inertia, body.inertia_inv, dt)
    return RigidBodyState(motor, rate, body.inertia, body.inertia_inv)


def rk4_step(body: RigidBodyState, dt: float) -> RigidBodyState:
    """Explicit RK4 on the rate, then a rotor step with the new rate."""
    motor, rate = explicit_rk4(body.motor, body.rate, body.inertia, body.inertia_inv, dt)
    return RigidBodyState(motor, rate, body.inertia, body.inertia_inv)


def rkmk4_step(body: RigidBodyState, dt: float) -> RigidBodyState:
    """Explicit 4th-order Munthe-Kaas: RK4 in the Lie algebra with the dexpinv correction."""
    motor, rate = explicit_rkmk4(body.motor, body.rate, body.inertia, body.inertia_inv, dt)
    return RigidBodyState(motor, rate, body.inertia, body.inertia_inv)


def simulate_tennis_racket(
    context: NumpyContext,
    dt: float,
    runtime: float,
    seed: int,
    step: Callable,
) -> list[RigidBodyState]:
    """Simulate rotation for all independent spin planes."""
    body = setup_tennis_racket_bodies(context, seed)
    n_steps = int(runtime / dt)

    states = [body]
    for _ in range(n_steps):
        body = step(body, dt)
        states.append(body)

    return states


def draw_trajectories(states: list[RigidBodyState], p: int, save_path: str) -> None:
    """Plot angular velocity trajectories for all axes."""
    trajectory = np.array([b.rate.kernel for b in states])
    init_body = states[0]
    energies = (0.5 * (init_body.inertia(init_body.rate) & init_body.rate)).kernel.ravel()
    nb = trajectory.shape[1]

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(nb, 1, figsize=(10, 2.0 * nb), squeeze=False, sharex=True)
    for i in range(nb):
        ax = axes[i, 0]
        ax.plot(trajectory[:, i, :])
        ax.set_ylabel(f"E ≈ {int(energies[i])}")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_title(f"Medial Axis Theorem ({p}D, {nb} Bivector Planes): Angular Velocities")
    axes[-1, 0].set_xlabel("Time step")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved {save_path}")
    plt.show()


def run_and_plot(
    p: int,
    dt: float,
    runtime: float,
    seed: int,
    step: Callable,
    save_path: str,
) -> None:
    """Run simulation and plot angular velocity trajectories for all axes."""
    context = NumpyContext(Algebra.from_pqr(p, 0, 0), dtype=np.float64)
    states = simulate_tennis_racket(context, dt, runtime, seed, step)
    draw_trajectories(states, p, save_path)


def draw_integrator_comparison(curves, dims: tuple[int, ...], dt: float, save_path: str) -> None:
    """Compare conservation errors after all integrations have finished."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(dims), figsize=(5.0 * len(dims), 4.0), squeeze=False, sharey=True)
    for p, name, momenta in curves:
        ax = axes[0, dims.index(p)]
        drift = np.linalg.norm(momenta - momenta[0], axis=-1) / np.linalg.norm(momenta[0], axis=-1)
        ax.semilogy(np.arange(len(drift)) * dt, drift.max(axis=1), label=name,
                    linestyle="--" if name == "rk4" else "-")
        ax.set_title(f"{p}D, {p * (p - 1) // 2} spin planes, dt = {dt}")
        ax.set_xlabel("Time"); ax.grid(True, alpha=0.3)
    axes[0, 0].set_ylabel("World momentum drift"); axes[0, 0].legend()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()


def run_integrator_comparison(
    dims: tuple[int, ...],
    dt: float,
    runtime: float,
    seed: int,
    save_path: str,
) -> None:
    """Compare world-momentum drift of the integrators, worst body per step, per dimension."""
    steppers = [
        ("verlet", verlet_step),
        ("rk4", rk4_step),
        ("rkmk4", rkmk4_step),
    ]
    curves = []
    for p in dims:
        context = NumpyContext(Algebra.from_pqr(p, 0, 0), dtype=np.float64)
        for label, step in steppers:
            states = simulate_tennis_racket(context, dt, runtime, seed, step)
            # World-frame angular momentum: L = motor >> inertia(rate)
            momenta = np.array([(b.motor >> b.inertia(b.rate)).kernel for b in states])
            curves.append((p, label, momenta))
    draw_integrator_comparison(curves, dims, dt, save_path)


if __name__ == "__main__":
    run_and_plot(
        p=4,
        dt=0.25,
        runtime=200.0,
        seed=42,
        step=verlet_step,
        save_path=str(PLOT_DIR / "tennis_racket.png"),
    )
    run_integrator_comparison(
        dims=(3, 4, 5),
        dt=0.25,
        runtime=100.0,
        seed=42,
        save_path=str(PLOT_DIR / "tennis_racket_integrators.png"),
    )

