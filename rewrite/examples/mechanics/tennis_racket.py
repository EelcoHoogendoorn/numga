"""Test the tennis racket theorem, or medial axis theorem.

Demonstrates the Dzhanibekov effect / intermediate axis theorem:
A rigid body with 3 distinct principal moments of inertia rotates stably
around its major and minor axes, but tumbles unstably around its intermediate
(medial) axis.
"""

from __future__ import annotations

import numpy as np

from numga.algebra import Algebra
from numga import NumpyContext
from examples.mechanics.rigid_body.core import Body
from examples.mechanics.rigid_body.lie_integrators import explicit_rk4, explicit_rkmk4
from examples import PLOT_DIR



def make_n_cube(n: int) -> np.ndarray:
    """Vertices of an n-dimensional hypercube centered at origin."""
    bits = ((np.arange(2**n)[:, None] & (1 << np.arange(n))) > 0)
    return 2.0 * bits - 1.0


def make_n_rect(n: int) -> np.ndarray:
    """Asymmetric rectangular cuboid with distinct axis lengths."""
    return make_n_cube(n) * (np.arange(n) + 1.0)


def setup_tennis_racket_bodies(
    context: NumpyContext = NumpyContext(Algebra.from_pqr(4, 0, 0), dtype=np.float64),
    seed: int = 42,
) -> tuple[Body, np.ndarray]:
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
    body = Body.from_point_cloud(points_batched)


    # Initial spin in each principal plane with a small perturbation
    rng = np.random.default_rng(seed)
    rates = np.eye(nb) + rng.normal(scale=1e-5, size=(nb, nb))
    body = body.copy(rate=context.multivector.bivector(rates))
    return body, body.kinetic_energy().kernel.ravel()


def verlet_step(body: Body, dt: float) -> Body:
    """The body's own XPBD Verlet step."""
    return body.integrate(dt)


def rk4_step(body: Body, dt: float) -> Body:
    """Explicit RK4 on the rate, then a rotor step with the new rate."""
    motor, rate = explicit_rk4(body.motor, body.rate, body.inertia, body.inertia_inv, dt)
    return body.copy(motor=motor, rate=rate)


def rkmk4_step(body: Body, dt: float) -> Body:
    """Explicit 4th-order Munthe-Kaas: RK4 in the Lie algebra with the dexpinv correction."""
    motor, rate = explicit_rkmk4(body.motor, body.rate, body.inertia, body.inertia_inv, dt)
    return body.copy(motor=motor, rate=rate)


INTEGRATORS = {"verlet": verlet_step, "rk4": rk4_step, "rkmk4": rkmk4_step}


def simulate_tennis_racket(
    context: NumpyContext = NumpyContext(Algebra.from_pqr(4, 0, 0), dtype=np.float64),
    dt: float = 0.25,
    runtime: float = 200.0,
    seed: int = 42,
    integrator: str = "verlet",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simulate rotation for all independent spin planes.

    Parameters
    ----------
    context : NumpyContext
        Geometric algebra context. Defaults to 4D Euclidean (Algebra.from_pqr(4, 0, 0)).
        Also works for p=2, 3, 5.
    dt : float
        Time step (default 0.25).
    runtime : float
        Total simulation duration (default 200.0).
    integrator : str
        One of INTEGRATORS: "verlet", "rk4", or "rkmk4".

    Returns
    -------
    trajectory : np.ndarray
        Array of shape `[n_steps, n_bodies, n_bivectors]` containing rates over time.
    energies : np.ndarray
        Initial kinetic energies of each body.
    energy_history : np.ndarray
        Array of shape `[n_steps, n_bodies]` containing kinetic energies over time.
    momentum_history : np.ndarray
        Array of shape `[n_steps, n_bodies, n_bivectors]` containing the world-frame
        angular momentum `motor >> inertia(rate)` over time. It is conserved exactly by
        the dynamics, and unlike the energy it sees how the motor is integrated.
    """
    body, energies = setup_tennis_racket_bodies(context, seed=seed)
    step = INTEGRATORS[integrator]
    n_steps = int(runtime / dt)
    states = []
    energy_history = []
    momentum_history = []

    for _ in range(n_steps):
        body = step(body, dt)
        states.append(body.rate.kernel)
        energy_history.append(body.kinetic_energy().kernel.ravel())
        momentum_history.append((body.motor >> body.inertia(body.rate)).kernel)

    return np.array(states), energies, np.array(energy_history), np.array(momentum_history)


def draw_trajectories(trajectory: np.ndarray, energies: np.ndarray, p: int, save_path: str) -> None:
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
    p: int = 4,
    dt: float = 0.25,
    runtime: float = 200.0,
    save_path: str = str(PLOT_DIR / "tennis_racket.png"),
) -> None:
    """Run simulation and plot angular velocity trajectories for all axes."""
    context = NumpyContext(Algebra.from_pqr(p, 0, 0), dtype=np.float64)
    trajectory, energies, _, _ = simulate_tennis_racket(context, dt=dt, runtime=runtime)
    draw_trajectories(trajectory, energies, p, save_path)


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
    dims: tuple[int, ...] = (3, 4, 5),
    dt: float = 0.25,
    runtime: float = 100.0,
    save_path: str = str(PLOT_DIR / "tennis_racket_integrators.png"),
) -> None:
    """Compare world-momentum drift of the integrators, worst body per step, per dimension."""
    curves = []
    for p in dims:
        context = NumpyContext(Algebra.from_pqr(p, 0, 0), dtype=np.float64)
        for name in INTEGRATORS:
            _, _, _, momenta = simulate_tennis_racket(context, dt=dt, runtime=runtime, integrator=name)
            curves.append((p, name, momenta))
    draw_integrator_comparison(curves, dims, dt, save_path)


if __name__ == "__main__":
    run_and_plot()
    run_integrator_comparison()

