"""Scenarios for the tennis racket theorem across Euclidean dimensions and Lie-group integrators."""

from __future__ import annotations

from collections.abc import Iterator

from numga import Algebra
from examples import instantiate

dimension = 4
step = 0.25
runtime = 200.0
seed = 42
compared_dimensions = (3, 4, 5)
comparison_runtime = 100.0
integrators = ("verlet", "rk4", "rkmk4")


# --- math -----------------------------------------------------------------------------
def spinning_racket(dimension: int, dt: float, runtime: float, seed: int):
    """Torque-free motion in `dimension` dimensions yields the rates and kinetic energies (`[n_steps + 1, n_planes]`)."""
    n_steps = int(runtime / dt)
    core = instantiate("examples.mechanics.tennis_racket.core", Algebra.from_pqr(dimension, 0, 0))
    motors, rates, inertia, inverse_inertia = core.racket(seed)
    _, step_rates = core.simulated_trajectory(motors, rates, inertia, inverse_inertia, core.lie.explicit_verlet, dt, n_steps)  # [n_steps + 1, ...] Motor+, Rate+
    return step_rates, core.lie.kinetic_energy(step_rates, inertia)


def integrator_comparison(dimensions: tuple[int, ...], dt: float, runtime: float, seed: int) -> Iterator:
    """Yields `(dimension, name, drift)` with world-momentum drift (`[n_steps + 1, n_planes] Scalar+`) per integrator."""
    n_steps = int(runtime / dt)
    for dim in dimensions:
        core = instantiate("examples.mechanics.tennis_racket.core", Algebra.from_pqr(dim, 0, 0))
        motors, rates, inertia, inverse_inertia = core.racket(seed)
        for name in integrators:
            integrator = getattr(core.lie, f"explicit_{name}")
            step_motors, step_rates = core.simulated_trajectory(motors, rates, inertia, inverse_inertia, integrator, dt, n_steps)
            yield dim, name, core.momentum_drift(step_motors, step_rates, inertia)


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_figure
    from examples.mechanics.tennis_racket import render

    rates, energies = spinning_racket(dimension, step, runtime, seed)
    save_figure(render.trajectories_figure(rates, energies), "tennis_racket")
    curves = list(integrator_comparison(compared_dimensions, step, comparison_runtime, seed))
    save_figure(render.integrator_comparison_figure(curves, step), "tennis_racket_integrators")

    # checks
    # Every body keeps its kinetic energy to step accuracy, and Munthe-Kaas keeps world momentum far better than Verlet or RK4.
    assert ((energies / energies[0] - 1).abs() < 1e-4).all()
    worst = {(dim, name): drift.max() for dim, name, drift in curves}
    for dim in compared_dimensions:
        assert bool(worst[dim, "rkmk4"] < 0.1 * worst[dim, "verlet"])
        assert bool(worst[dim, "rkmk4"] < 0.1 * worst[dim, "rk4"])


if __name__ == "__main__":
    main()
