"""A cooling coil and a reheater bring warm humid air to a comfortable supply condition."""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import concatenate, stack
from examples.thermodynamics.property_charts import core

pressure = 101.325
inlet_temperature, inlet_relative_humidity = 30.0, 0.7
coil_temperature, supply_temperature = 12.0, 22.0
n_steps, frame_duration_ms = 52, 40


# --- math -----------------------------------------------------------------------------
def relative_humidity_of(states: core.State) -> core.Scalar:
    """Relative humidity of moist air states at atmospheric pressure."""
    return core.relative_humidity(states, pressure)


def conditioning(n_steps: int) -> tuple[core.State, core.State, core.State, tuple[core.Scalar, ...]]:
    """Air path across the coil and reheater, inlet and coil-outlet states, constrained tangent directions there, and
    the cooling, condensed water, reheating, and supply relative humidity."""
    fractions = np.linspace(0, 1, n_steps)
    inlet = core.air_state(inlet_temperature, inlet_relative_humidity, pressure)                # [] State-
    cooled = core.cooling_path(inlet, coil_temperature, pressure, fractions)                    # [n_steps] State-
    outlet = cooled[-1]                                                                         # [] State-
    reheated = core.reheating_path(outlet, supply_temperature, fractions)                       # [n_steps] State-
    path = concatenate([cooled, reheated[1:]])                                                  # [2 * n_steps - 1] State-
    cooling, condensate = core.coil_balance(inlet, outlet)                                      # [] Scalar+, Scalar+
    reheating = core.enthalpy(reheated[-1]) - core.enthalpy(outlet)                             # [] Scalar+
    supply = core.relative_humidity(reheated[-1], pressure)                                     # [] Scalar+
    # Unsaturated air at the inlet fixes water content (`w`), while saturated air at the outlet fixes relative humidity.
    outlet_slope = core.relative_humidity_slope(outlet, pressure)                               # [] Potential+
    directions = stack((core.constrained_direction(core.mv.w),
                        core.constrained_direction(outlet_slope)))                              # [2] State-
    return path, stack((inlet, outlet)), directions, (cooling, condensate, reheating, supply)


def air_states(path: core.State) -> Iterator[core.State]:
    """Yields each air state along the conditioning path."""
    for index in range(path.shape[0]):
        yield path[index]


def main() -> None:
    """Save the psychrometric chart figure and the air-conditioning animation."""
    from examples.util.animation import save_animation, save_figure
    from examples.thermodynamics.property_charts import render

    path, key_states, directions, (cooling, condensate, reheating, _) = conditioning(n_steps)
    save_figure(render.draw_chart(relative_humidity_of, core.enthalpy, key_states, directions),
                "air_conditioning_chart")
    save_animation(
        render.conditioning(relative_humidity_of, core.enthalpy, path, key_states, directions, air_states(path)),
        "air_conditioning", frame_duration_ms,
    )

    # checks
    inlet, outlet = key_states[0], key_states[1]
    supply_state = path[-1]
    outlet_slope = core.relative_humidity_slope(outlet, pressure)
    np.testing.assert_allclose(((core.mv.w & inlet) - (core.mv.w & supply_state) - condensate).kernel, 0, atol=1e-12)
    np.testing.assert_allclose(
        (core.enthalpy(inlet) - core.enthalpy(supply_state) - cooling
         - condensate * core.liquid_heat * coil_temperature + reheating).kernel,
        0, atol=1e-10,
    )
    assert np.all(relative_humidity_of(path) <= 1 + 1e-10)
    np.testing.assert_allclose(((core.mv.t & directions) - 1).kernel, 0, atol=1e-10)
    np.testing.assert_allclose((core.mv.w & directions[0]).kernel, 0, atol=1e-10)
    np.testing.assert_allclose((outlet_slope & directions[1]).kernel, 0, atol=1e-10)


if __name__ == "__main__":
    main()
