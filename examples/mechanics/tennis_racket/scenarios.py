"""Scenes for the tennis racket theorem: one spinning racket, and a comparison of integrators."""

from __future__ import annotations

from numga import Algebra
from examples import instantiate


def spinning_racket(dimension: int, dt: float, runtime: float, seed: int):
    """Rates of a racket spun in every bivector plane, stacked over time, and their energies."""
    core = instantiate("examples.mechanics.tennis_racket.core", Algebra.from_pqr(dimension, 0, 0))
    body = core.racket(seed)
    _, rates = core.simulate(body, core.lie.explicit_verlet, dt, int(runtime / dt))
    return rates, core.lie.kinetic_energy(body.rate, body.inertia)


def integrator_comparison(dimensions: tuple[int, ...], dt: float, runtime: float, seed: int):
    """World-momentum drift of three integrators, per dimension, step and body."""
    curves = []
    for dimension in dimensions:
        core = instantiate("examples.mechanics.tennis_racket.core", Algebra.from_pqr(dimension, 0, 0))
        body = core.racket(seed)
        for name, step in (("verlet", core.lie.explicit_verlet),
                           ("rk4", core.lie.explicit_rk4),
                           ("rkmk4", core.lie.explicit_rkmk4)):
            motors, rates = core.simulate(body, step, dt, int(runtime / dt))
            curves.append((dimension, name, core.momentum_drift(motors, rates, body.inertia)))
    return curves


if __name__ == "__main__":
    from examples.animation import save_figure
    from examples.mechanics.tennis_racket import render

    save_figure(render.draw_trajectories(*spinning_racket(4, 0.25, 200.0, 42)), "tennis_racket")
    curves = integrator_comparison((3, 4, 5), 0.25, 100.0, 42)
    save_figure(render.draw_integrator_comparison(curves, 0.25), "tennis_racket_integrators")
