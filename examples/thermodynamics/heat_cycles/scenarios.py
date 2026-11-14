"""A reversible Carnot engine heated and cooled between two thermal reservoirs."""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from examples.thermodynamics.heat_cycles import core
from numga.topology import Topology

capacity = 1.5
hot, cold, entropy_span = 1.5, 1.0, 1.0
n_steps, frame_duration_ms = 28, 40


# --- math -----------------------------------------------------------------------------
def corners() -> tuple[Topology, core.Scalar, core.Scalar]:
    """Corner loop topology and corner entropies and temperatures for the four Carnot strokes."""
    corner_entropy = core.mv.scalar([[0], [entropy_span], [entropy_span], [0]]).field()   # Scalar+[V]
    corner_temperature = core.mv.scalar([[hot], [hot], [cold], [cold]]).field()           # Scalar+[V]
    return Topology.loops([corner_entropy.sites.count]), corner_entropy, corner_temperature


def cycle(n_steps: int) -> tuple[core.State, core.Scalar, core.Scalar]:
    """Closed loop of states across all strokes, with midpoint heat absorbed and work done along each step."""
    around, corner_entropy, corner_temperature = corners()
    n_corners = around.n_vertices
    starts = core.mv.scalar(np.tile(np.arange(n_steps) / n_steps, n_corners)[:, None]).field()            # Scalar+[V_samples]
    middles = core.mv.scalar(np.tile((np.arange(n_steps) + 0.5) / n_steps, n_corners)[:, None]).field()   # Scalar+[V_samples]
    states, _ = core.stroke_samples(starts, around, corner_entropy, corner_temperature, capacity, n_steps)
    mid_states, tangents = core.stroke_samples(middles, around, corner_entropy, corner_temperature, capacity, n_steps)
    heat_rate, work_rate = core.exchanges(mid_states, tangents, capacity)                 # Scalar+[V_samples], Scalar+[V_samples]
    return states, heat_rate / n_steps, work_rate / n_steps


def temperature_of(states: core.State) -> core.Scalar:
    """Absolute temperature at each gas state."""
    return core.temperature_of(states, capacity)


def pressure_of(states: core.State) -> core.Scalar:
    """Pressure at each gas state."""
    return core.pressure_of(states, capacity)


def cycle_states(states: core.State) -> Iterator[core.State]:
    """Yield each gas state in turn around the cycle."""
    for state in states.batch():
        yield state


def main() -> None:
    from examples.thermodynamics.heat_cycles import render
    from examples.util.animation import save_animation, save_figure

    states, _, _ = cycle(n_steps)
    save_figure(render.draw_cycle(temperature_of, pressure_of, states), "thermodynamics_heat_cycle")
    save_animation(render.animate_cycle(temperature_of, pressure_of, states, cycle_states(states)),
                   "thermodynamics_heat_cycle", frame_duration_ms)


if __name__ == "__main__":
    main()
