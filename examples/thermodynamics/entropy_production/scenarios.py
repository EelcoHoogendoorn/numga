"""Two insulated bodies at different temperatures exchange heat until their temperatures equalize."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import stack
from examples.thermodynamics.entropy_production import core

capacity_a, capacity_b, conductance = 1.0, 2.0, 0.6
initial_temperature_a, initial_temperature_b = 2.8, 0.6

initial_state = (
    core.mv.a.dual() * (capacity_a * initial_temperature_a)
    + core.mv.b.dual() * (capacity_b * initial_temperature_b)
)                                                                                 # [] State-
relaxation_time = 1 / (conductance * (1 / capacity_a + 1 / capacity_b))


# --- math -----------------------------------------------------------------------------
def entropy_of(states: core.State) -> core.Scalar:
    """Total entropy of the two-body pair at each state."""
    return core.entropy(states, capacity_a, capacity_b)


def entropy_potential_of(states: core.State) -> core.Potential:
    """Entropic potential of the two-body pair at each state."""
    return core.entropy_potential(states, capacity_a, capacity_b)


def temperatures_of(states: core.State) -> core.Scalar:
    """Temperatures of the two bodies stacked along their trailing axis."""
    return stack(core.temperatures(states, capacity_a, capacity_b), axis=-1)


def exchange(n_frames: int, n_relaxation_times: float) -> Iterator[tuple[core.State, float]]:
    """Yield the state and elapsed time at each frame over `n_relaxation_times` relaxation times."""
    dt = n_relaxation_times * relaxation_time / (n_frames - 1)
    state = initial_state
    for frame in range(n_frames):
        yield state, frame * dt
        state = core.advance(state, capacity_a, capacity_b, conductance, dt)


def main() -> None:
    from examples.thermodynamics.entropy_production import render
    from examples.util.animation import save_animation, save_figure

    n_frames, n_relaxation_times, frame_duration_ms = 135, 5, 40
    capacities = np.array([capacity_a, capacity_b])
    frames = list(exchange(n_frames, n_relaxation_times))
    first_state, _ = frames[0]                                                    # [] State-
    save_figure(
        render.draw(entropy_of, entropy_potential_of, temperatures_of, core.transfer, capacities, first_state),
        "entropy_production",
    )
    save_animation(
        render.exchange(entropy_of, entropy_potential_of, temperatures_of, core.transfer, capacities, iter(frames)),
        "entropy_production",
        frame_duration_ms,
    )

    # checks
    states = stack([state for state, _ in frames])                                # [n_frames] State-
    entropies = entropy_of(states)                                                # [n_frames] Scalar+
    np.testing.assert_allclose(
        ((core.energy_potential & states) - (core.energy_potential & initial_state)).kernel, 0, atol=1e-6
    )
    assert np.all(entropies[1:] - entropies[:-1] >= -1e-6)
    np.testing.assert_allclose(
        core.mobility(initial_state, capacity_a, capacity_b, conductance)(core.energy_potential).kernel, 0, atol=1e-6
    )
    np.testing.assert_allclose((temperatures_of(states[-1]) - 4 / 3).kernel, 0, atol=1e-2)


if __name__ == "__main__":
    main()
