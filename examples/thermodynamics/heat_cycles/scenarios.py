"""A reversible engine heated and cooled by two reservoirs."""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import concatenate
from examples.thermodynamics.heat_cycles import core

CAPACITY = 1.5
HOT, COLD, ENTROPY_SPAN = 1.5, 1.0, 1.0


# --- math -----------------------------------------------------------------------------
def corners() -> tuple[core.Scalar, core.Scalar]:
    """The cycle's corners in entropy and temperature `[5]`, the first again at the end: heated at the hot
    temperature, expanded insulated, cooled at the cold temperature, compressed insulated."""
    return (core.mv.scalar([[0], [ENTROPY_SPAN], [ENTROPY_SPAN], [0], [0]]),
            core.mv.scalar([[HOT], [HOT], [COLD], [COLD], [HOT]]))


def cycle(n_steps: int) -> tuple[core.State, core.Scalar, core.Scalar]:
    """The cycle's states, every stroke in steps, the first state again at the end `[4 * n_steps + 1]`; and the heat
    taken in and the work done up to each, by the midpoint rule along the strokes' tangents."""
    entropies, temperatures = corners()
    along = lambda fraction: core.strokes(fraction, entropies[:, None], temperatures[:, None], CAPACITY)
    ends = core.mv.scalar(np.broadcast_to(np.arange(1, n_steps + 1) / n_steps, (4, n_steps))[..., None])
    middles = core.mv.scalar(np.broadcast_to((np.arange(n_steps) + 0.5) / n_steps, (4, n_steps))[..., None])
    # The tangent at each middle: the strokes' derivative there bound to a whole stroke's fraction.
    tangents = core.derivative(along)(middles)(core.mv.scalar([1]))                  # [strokes, steps] State
    heat_rate, work_rate = core.exchanges(along(middles), tangents, CAPACITY)
    zero = core.mv.scalar([[0]])
    states = concatenate([along(core.mv.scalar([[0]]))[0, :1], along(ends).reshape(-1)])
    heat = concatenate([zero, (heat_rate / n_steps).reshape(-1).cumsum(0)])
    work = concatenate([zero, (work_rate / n_steps).reshape(-1).cumsum(0)])
    return states, heat, work


def temperature_of(states: core.State) -> core.Scalar:
    return core.derivative(lambda state: core.energy(state, CAPACITY))(states)(core.mv.s.dual())


def pressure_of(states: core.State) -> core.Scalar:
    return -core.derivative(lambda state: core.energy(state, CAPACITY))(states)(core.mv.v.dual())


def running(states: core.State) -> Iterator[core.State]:
    """The gas's state at each step around the cycle."""
    for index in range(states.shape[0]):
        yield states[index]


if __name__ == "__main__":
    from examples.util.animation import save_animation, save_figure
    from examples.thermodynamics.heat_cycles import render

    states, heat, work = cycle(16)
    save_figure(render.draw_cycle(temperature_of, pressure_of, states), "thermodynamics_heat_cycle")
    save_animation(render.animate_cycle(temperature_of, pressure_of, states, running(states)),
                   "thermodynamics_heat_cycle", 70)
