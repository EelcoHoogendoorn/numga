"""Equal heat taken in by a cylinder latched at fixed volume and by one under a weight at fixed pressure."""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from examples.thermodynamics.state_space import core

THERMAL_CAPACITY = 1.5


# --- math -----------------------------------------------------------------------------
def temperature_of(states: core.State) -> core.Scalar:
    return core.response(states, THERMAL_CAPACITY)[0]


def pressure_of(states: core.State) -> core.Scalar:
    return core.response(states, THERMAL_CAPACITY)[1]


def initial() -> core.State:
    """Both cylinders at the reference state: unit volume and temperature, entropy measured from there."""
    return (core.mv.s.dual() * 0 + core.mv.v.dual())[None].broadcast_to((2,))


def equal_heating(total_heat: float, n_steps: int) -> Iterator[tuple[core.State, core.State]]:
    """The two cylinders' states `[2]`, latched then loaded, and their heating directions, as equal slices of heat come
    in."""
    states = initial()
    for step in range(n_steps + 1):
        directions, _ = core.heating_direction(states, core.held(states, THERMAL_CAPACITY), THERMAL_CAPACITY)
        yield states, directions
        states = core.heated(states, total_heat / n_steps, THERMAL_CAPACITY)


if __name__ == "__main__":
    from examples.util.animation import save_animation, save_figure
    from examples.thermodynamics.state_space import render

    first, directions = next(equal_heating(2.4, 48))
    save_figure(render.draw_directions(temperature_of, pressure_of, first, directions), "thermodynamics_heating_charts")
    save_animation(render.heating(temperature_of, pressure_of, equal_heating(2.4, 48)), "thermodynamics_heating", 80)
