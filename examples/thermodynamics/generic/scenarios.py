"""A mass released on a spring, its motion damped by friction that warms a thermal store."""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from examples.thermodynamics.generic import core

MASS, STIFFNESS, FRICTION = 1.0, 4.0, 0.35
THERMAL_CAPACITY, REFERENCE_TEMPERATURE = 2.0, 1.0


# --- math -----------------------------------------------------------------------------
def initial() -> core.State:
    """Displaced by two, at rest, the store at the reference temperature."""
    return core.mv.q.dual() * 2.0 + core.mv.p.dual() * 0.0 + core.mv.s.dual() * 0.0


def energies_of(state: core.State) -> tuple[core.Scalar, core.Scalar, core.Scalar]:
    return (core.kinetic_energy(state, MASS), core.potential_energy(state, STIFFNESS),
            core.internal_energy(state, THERMAL_CAPACITY, REFERENCE_TEMPERATURE))


def surface_entropy_of(state: core.State) -> core.Scalar:
    """The store's entropy when the total energy is the start's: at fixed energy, the entropy as a function of the
    position and momentum, all the energy they leave over held by the store."""
    total = core.total_energy(initial(), MASS, STIFFNESS, THERMAL_CAPACITY, REFERENCE_TEMPERATURE)
    left = total - core.kinetic_energy(state, MASS) - core.potential_energy(state, STIFFNESS)
    return THERMAL_CAPACITY * (left / (THERMAL_CAPACITY * REFERENCE_TEMPERATURE)).log()


def rates_of(state: core.State) -> tuple[core.State, core.State]:
    return core.rates(state, MASS, STIFFNESS, FRICTION, THERMAL_CAPACITY, REFERENCE_TEMPERATURE)


def release(n_frames: int, steps_per_frame: int, dt: float) -> Iterator[core.State]:
    """The state at each frame, so many steps of the combined motion apart."""
    state = initial()
    for _ in range(n_frames):
        yield state
        for _ in range(steps_per_frame):
            state = core.advance(state, MASS, STIFFNESS, FRICTION, THERMAL_CAPACITY, REFERENCE_TEMPERATURE, dt)


if __name__ == "__main__":
    from examples.util.animation import save_animation, save_figure
    from examples.thermodynamics.generic import render

    save_figure(render.draw(energies_of, surface_entropy_of, rates_of, initial()), "generic_thermodynamics")
    save_animation(render.damping(energies_of, surface_entropy_of, rates_of, release(150, 2, 0.04), 2 * 0.04),
                   "generic_thermodynamics", 40)
