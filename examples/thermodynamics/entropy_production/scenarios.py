"""Two insulated bodies at different temperatures share their energy until their temperatures meet."""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import stack
from examples.thermodynamics.entropy_production import core

capacity_a, capacity_b, conductance = 1.0, 2.0, 0.6
initial_temperature_a, initial_temperature_b = 2.8, 0.6

initial_state = (core.mv.a.dual() * (capacity_a * initial_temperature_a)
                 + core.mv.b.dual() * (capacity_b * initial_temperature_b))          # [] State-
relaxation_time = 1 / (conductance * (1 / capacity_a + 1 / capacity_b))


# --- math -----------------------------------------------------------------------------
def entropy_of(states: core.State) -> core.Scalar:
    return core.entropy(states, capacity_a, capacity_b)


def temperatures_of(states: core.State) -> core.Scalar:
    """The two bodies' temperatures `[..., bodies]`."""
    return stack(core.temperatures(states, capacity_a, capacity_b), axis=-1)


def exchange(n_frames: int, n_relaxation_times: float) -> Iterator[tuple[core.State, core.State, float]]:
    """The two bodies' state, its heat flow and the time at each frame, over a number of relaxation times."""
    dt = n_relaxation_times * relaxation_time / (n_frames - 1)
    state = initial_state
    for frame in range(n_frames):
        yield state, core.heat_flow(state, capacity_a, capacity_b, conductance), frame * dt
        state = core.advance(state, capacity_a, capacity_b, conductance, dt)


if __name__ == "__main__":
    from examples.util.animation import save_animation, save_figure
    from examples.thermodynamics.entropy_production import render

    n_frames, n_relaxation_times, frame_duration_ms = 72, 5, 75
    capacities = np.array([capacity_a, capacity_b])
    first_state, first_flow, _ = next(exchange(n_frames, n_relaxation_times))
    save_figure(render.draw(entropy_of, temperatures_of, core.transfer, capacities, first_state, first_flow),
                "entropy_production")
    save_animation(render.exchange(entropy_of, temperatures_of, core.transfer, capacities,
                                   exchange(n_frames, n_relaxation_times)), "entropy_production", frame_duration_ms)
