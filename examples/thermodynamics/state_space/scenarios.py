"""Equal heat taken in by a cylinder latched at fixed volume and by one under a weight at fixed pressure."""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from examples.thermodynamics.state_space import core

thermal_capacity = 1.5


# --- math -----------------------------------------------------------------------------
def temperature_of(states: core.State) -> core.Scalar:
    return core.response(states, thermal_capacity)[0]


def pressure_of(states: core.State) -> core.Scalar:
    return core.response(states, thermal_capacity)[1]


def initial() -> core.State:
    """Both cylinders at the reference state: unit volume and temperature, entropy measured from there."""
    return (core.mv.s.dual() * 0 + core.mv.v.dual())[None].broadcast_to((2,))


def equal_heating(total_heat: float, n_steps: int) -> Iterator[tuple[core.State, core.State, float]]:
    """The two cylinders' states `[cylinders]`, latched then loaded, their heating directions, and the heat each has
    taken in, as equal slices of heat come in."""
    states = initial()
    for step in range(n_steps + 1):
        directions, _ = core.heating_direction(states, core.held(states, thermal_capacity), thermal_capacity)
        yield states, directions, step * total_heat / n_steps
        states = core.heated(states, total_heat / n_steps, thermal_capacity)


if __name__ == "__main__":
    from examples.util.animation import save_animation, save_figure
    from examples.thermodynamics.state_space import render

    total_heat, n_steps, frame_duration_ms = 2.4, 97, 40
    first, directions, _ = next(equal_heating(total_heat, n_steps))
    save_figure(render.draw_directions(temperature_of, pressure_of, first, directions), "thermodynamics_heating_charts")
    save_animation(render.heating(temperature_of, pressure_of, equal_heating(total_heat, n_steps)),
                   "thermodynamics_heating", frame_duration_ms)
