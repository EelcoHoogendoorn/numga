"""A cooling coil and a reheater bring warm humid air to a supply condition."""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import concatenate, stack
from examples.thermodynamics.property_charts import core

PRESSURE = 101.325
INLET_TEMPERATURE, INLET_HUMIDITY = 30.0, 0.7
COIL_TEMPERATURE, SUPPLY_TEMPERATURE = 12.0, 22.0


# --- math -----------------------------------------------------------------------------
def relative_humidity_of(states: core.State) -> core.Scalar:
    return core.relative_humidity(states, PRESSURE)


def conditioning(n_steps: int) -> tuple[core.State, core.State, core.State, tuple[core.Scalar, ...]]:
    """The air's path through the coil and the reheater; its state at the inlet and at the coil's outlet; the
    directions it moves in there, holding its water content and holding its relative humidity; and the cooling, the
    water collected, the reheating and the supply's relative humidity."""
    fraction = np.linspace(0, 1, n_steps)
    initial = core.air_state(INLET_TEMPERATURE, INLET_HUMIDITY, PRESSURE)
    cooled = core.cooling_path(initial, COIL_TEMPERATURE, PRESSURE, fraction)
    reheated = core.reheating_path(cooled[-1], SUPPLY_TEMPERATURE, fraction)
    path = concatenate([cooled, reheated[1:]])
    cooling, condensate = core.coil_balance(initial, cooled[-1])
    reheating = core.enthalpy(reheated[-1]) - core.enthalpy(cooled[-1])
    supply = core.relative_humidity(reheated[-1], PRESSURE)
    saturated = core.derivative(relative_humidity_of)(cooled[-1])
    directions = stack((core.constrained_direction(initial, core.humidity),
                        core.constrained_direction(cooled[-1], saturated)))
    return path, stack((initial, cooled[-1])), directions, (cooling, condensate, reheating, supply)


def passing(path: core.State) -> Iterator[core.State]:
    """The air's state at each step along its path."""
    for index in range(path.shape[0]):
        yield path[index]


if __name__ == '__main__':
    from examples.util.animation import save_animation, save_figure
    from examples.thermodynamics.property_charts import render

    path, states, directions, _ = conditioning(32)
    save_figure(render.draw_chart(relative_humidity_of, core.enthalpy, states, directions), 'air_conditioning_chart')
    save_animation(render.conditioning(relative_humidity_of, core.enthalpy, path, states, directions, passing(path)),
                   'air_conditioning', 65)
