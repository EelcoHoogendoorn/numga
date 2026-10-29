"""A cooling coil and a reheater bring warm humid air to a supply condition."""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import concatenate, stack
from numga.backend.jax import derivative
from examples.thermodynamics.property_charts import core

pressure = 101.325
inlet_temperature, inlet_relative_humidity = 30.0, 0.7
coil_temperature, supply_temperature = 12.0, 22.0


# --- math -----------------------------------------------------------------------------
def relative_humidity_of(states: core.State) -> core.Scalar:
    return core.relative_humidity(states, pressure)


def conditioning(n_steps: int) -> tuple[core.State, core.State, core.State, tuple[core.Scalar, ...]]:
    """The air's path through the coil and the reheater; its state at the inlet and at the coil's outlet; the
    directions it moves in there, holding its water content and holding its relative humidity; and the cooling, the
    water collected, the reheating and the supply's relative humidity."""
    fractions = np.linspace(0, 1, n_steps)
    inlet = core.air_state(inlet_temperature, inlet_relative_humidity, pressure)                # [] State-
    cooled = core.cooling_path(inlet, coil_temperature, pressure, fractions)                   # [steps] State-
    outlet = cooled[-1]                                                                         # [] State-
    reheated = core.reheating_path(outlet, supply_temperature, fractions)                      # [steps] State-
    path = concatenate([cooled, reheated[1:]])                                                  # [2 * steps - 1] State-
    cooling, condensate = core.coil_balance(inlet, outlet)                                      # [] Scalar+, Scalar+
    reheating = core.enthalpy(reheated[-1]) - core.enthalpy(outlet)                             # [] Scalar+
    supply = core.relative_humidity(reheated[-1], pressure)                                     # [] Scalar+
    # Saturated air holds its relative humidity at one: the reading it holds is the relative humidity's change.
    relative_humidity_change = derivative(relative_humidity_of)(outlet)                         # [] Scalar+ <- State-
    directions = stack((core.constrained_direction(inlet, core.humidity),
                        core.constrained_direction(outlet, relative_humidity_change)))         # [stages] State-
    return path, stack((inlet, outlet)), directions, (cooling, condensate, reheating, supply)


def passing(path: core.State) -> Iterator[core.State]:
    """The air's state at each step along its path."""
    for index in range(path.shape[0]):
        yield path[index]


if __name__ == '__main__':
    from examples.util.animation import save_animation, save_figure
    from examples.thermodynamics.property_charts import render

    n_steps, frame_duration_ms = 52, 40
    path, states, directions, _ = conditioning(n_steps)
    save_figure(render.draw_chart(relative_humidity_of, core.enthalpy, states, directions), 'air_conditioning_chart')
    save_animation(render.conditioning(relative_humidity_of, core.enthalpy, path, states, directions, passing(path)),
                   'air_conditioning', frame_duration_ms)
