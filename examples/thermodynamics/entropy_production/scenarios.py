"""Two insulated bodies at different temperatures share their energy until their temperatures meet."""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from examples.thermodynamics.entropy_production import core

CAPACITY_A, CAPACITY_B, CONDUCTANCE = 1.0, 2.0, 0.6
TEMPERATURE_A, TEMPERATURE_B = 2.8, 0.6


# --- math -----------------------------------------------------------------------------
def initial() -> core.State:
    return core.mv.a.dual() * (CAPACITY_A * TEMPERATURE_A) + core.mv.b.dual() * (CAPACITY_B * TEMPERATURE_B)


def entropy_of(states: core.State) -> core.Scalar:
    return core.entropy(states, CAPACITY_A, CAPACITY_B)


def temperatures_of(states: core.State) -> tuple[core.Scalar, core.Scalar]:
    return core.temperatures(states, CAPACITY_A, CAPACITY_B)


def relaxation_time() -> float:
    return 1 / (CONDUCTANCE * (1 / CAPACITY_A + 1 / CAPACITY_B))


def exchange(n_frames: int, n_relaxation_times: float) -> Iterator[tuple[core.State, core.State, float]]:
    """The two bodies' state, its heat flow and the time at each frame, over a number of relaxation times."""
    dt = n_relaxation_times * relaxation_time() / (n_frames - 1)
    state = initial()
    for frame in range(n_frames):
        yield state, core.heat_flow(state, CAPACITY_A, CAPACITY_B, CONDUCTANCE), frame * dt
        state = core.advance(state, CAPACITY_A, CAPACITY_B, CONDUCTANCE, dt)


if __name__ == "__main__":
    from examples.util.animation import save_animation, save_figure
    from examples.thermodynamics.entropy_production import render

    capacities = np.array([CAPACITY_A, CAPACITY_B])
    save_figure(render.draw(entropy_of, temperatures_of, core.transfer, capacities, *next(exchange(2, 5))[:2]),
                "entropy_production")
    save_animation(render.exchange(entropy_of, temperatures_of, core.transfer, capacities, exchange(72, 5)),
                   "entropy_production", 75)
