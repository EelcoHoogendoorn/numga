"""A reversible engine heated and cooled by two reservoirs."""
from __future__ import annotations

from collections.abc import Iterator

import jax
import numpy as np

from examples.thermodynamics.heat_cycles import core

capacity = 1.5
hot, cold, entropy_span = 1.5, 1.0, 1.0


# --- math -----------------------------------------------------------------------------
def corners() -> tuple[core.Scalar, core.Scalar]:
    """The cycle's corners in entropy and temperature `[corners]`, the strokes running from each to the next and from
    the last back to the first: heated at the hot temperature, expanded insulated, cooled at the cold temperature,
    compressed insulated."""
    return (core.mv.scalar([[0], [entropy_span], [entropy_span], [0]]),
            core.mv.scalar([[hot], [hot], [cold], [cold]]))


def cycle(n_steps: int) -> tuple[core.State, core.Scalar, core.Scalar]:
    """The cycle's states, every stroke in steps, a closed loop `[strokes * steps]`; and the heat taken in and the
    work done along each step, from each state to the next and from the last back to the first, by the midpoint rule
    along the strokes' tangents."""
    entropies, temperatures = corners()
    strokes = lambda fraction: core.strokes(fraction, entropies, temperatures, capacity)
    starts = core.mv.scalar(np.broadcast_to(np.arange(n_steps) / n_steps, (4, n_steps))[..., None])
    middles = core.mv.scalar(np.broadcast_to((np.arange(n_steps) + 0.5) / n_steps, (4, n_steps))[..., None])
    # The tangent at each middle: the strokes' derivative there bound to a whole stroke's fraction.
    tangents = core.derivative(strokes)(middles)(core.mv.scalar([1]))                # [strokes, steps] State-
    heat_rate, work_rate = core.exchanges(strokes(middles), tangents, capacity)
    return strokes(starts).reshape(-1), (heat_rate / n_steps).reshape(-1), (work_rate / n_steps).reshape(-1)


@jax.jit
def temperature_of(states: core.State) -> core.Scalar:
    return core.derivative(lambda state: core.energy(state, capacity))(states)(core.mv.s.dual())


@jax.jit
def pressure_of(states: core.State) -> core.Scalar:
    return -core.derivative(lambda state: core.energy(state, capacity))(states)(core.mv.v.dual())


def running(states: core.State) -> Iterator[core.State]:
    """The gas's state at each step around the cycle."""
    for index in range(states.shape[0]):
        yield states[index]


if __name__ == "__main__":
    from examples.util.animation import save_animation, save_figure
    from examples.thermodynamics.heat_cycles import render

    n_steps, frame_duration_ms = 28, 40
    states, _, _ = cycle(n_steps)
    save_figure(render.draw_cycle(temperature_of, pressure_of, states), "thermodynamics_heat_cycle")
    save_animation(render.animate_cycle(temperature_of, pressure_of, states, running(states)),
                   "thermodynamics_heat_cycle", frame_duration_ms)
