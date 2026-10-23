"""Scenes for the tennis racket theorem: one racket spun in every principal plane, and a comparison of
integrators by how well they keep the angular momentum in the world."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import Algebra
from examples import instantiate

# The racket's dimension, the time step and run time, and the seed of its small disturbances.
DIMENSION = 4
STEP = 0.25
RUNTIME = 200.0
SEED = 42
# The dimensions and run time of the comparison, and the integrators compared.
COMPARED_DIMENSIONS = (3, 4, 5)
COMPARISON_RUNTIME = 100.0
INTEGRATORS = ("verlet", "rk4", "rkmk4")


# --- math -----------------------------------------------------------------------------
def racket_core(dimension: int):
    """The racket's mathematics in the Euclidean algebra of the given dimension."""
    return instantiate("examples.mechanics.tennis_racket.core", Algebra.from_pqr(dimension, 0, 0))


def spinning_racket(dimension: int, dt: float, runtime: float, seed: int):
    """The rates of a racket spun in every principal plane, `[steps + 1, bodies]`, and their kinetic
    energies."""
    core = racket_core(dimension)
    body = core.racket(seed)
    _, rates = core.simulate(body, core.lie.explicit_verlet, dt, int(runtime / dt))
    return rates, core.lie.kinetic_energy(rates, body.inertia)


def integrator_comparison(dimensions: tuple[int, ...], dt: float, runtime: float, seed: int) -> Iterator:
    """Each integrator's drift of the world angular momentum, per dimension, `[steps + 1, bodies]`."""
    for dimension in dimensions:
        core = racket_core(dimension)
        body = core.racket(seed)
        for name in INTEGRATORS:
            motors, rates = core.simulate(body, getattr(core.lie, f"explicit_{name}"), dt, int(runtime / dt))
            yield dimension, name, core.momentum_drift(motors, rates, body.inertia)


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_figure
    from examples.mechanics.tennis_racket import render

    rates, energies = spinning_racket(DIMENSION, STEP, RUNTIME, SEED)
    save_figure(render.draw_trajectories(rates, energies), "tennis_racket")
    curves = list(integrator_comparison(COMPARED_DIMENSIONS, STEP, COMPARISON_RUNTIME, SEED))
    save_figure(render.draw_integrator_comparison(curves, STEP), "tennis_racket_integrators")

    # checks
    # Every body keeps its kinetic energy while it tumbles, to the accuracy of the steps; in every dimension
    # the Lie-group integrator keeps the world angular momentum far better than the others.
    np.testing.assert_allclose((energies / energies[0]).to_array(), 1.0, atol=1e-4)
    worst = {(dimension, name): drift.to_array().max() for dimension, name, drift in curves}
    for dimension in COMPARED_DIMENSIONS:
        assert worst[dimension, "rkmk4"] < 0.1 * min(worst[dimension, "verlet"], worst[dimension, "rk4"])


if __name__ == "__main__":
    main()
