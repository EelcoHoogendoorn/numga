"""A mass released on a spring, its motion damped by friction that warms a thermal store."""
from __future__ import annotations

from collections.abc import Iterator

from examples.thermodynamics.generic import core

mass, stiffness, friction = 1.0, 4.0, 0.35
thermal_capacity, reference_temperature = 2.0, 1.0
displacement = 2.0

# The mass starts displaced and at rest, the store at the reference temperature.
start = core.mv.q.dual() * displacement + core.mv.p.dual() * 0.0 + core.mv.s.dual() * 0.0          # [] State-
start_energy = core.total_energy(start, mass, stiffness, thermal_capacity, reference_temperature)  # [] Scalar+


# --- math -----------------------------------------------------------------------------
def energies_of(state: core.State) -> tuple[core.Scalar, core.Scalar, core.Scalar]:
    return (core.kinetic_energy(state, mass), core.potential_energy(state, stiffness),
            core.internal_energy(state, thermal_capacity, reference_temperature))


def surface_entropy_of(state: core.State) -> core.Scalar:
    """The store's entropy when the total energy is the start's: at fixed energy, the entropy as a function of the
    position and momentum, all the energy they leave over held by the store."""
    internal = start_energy - core.kinetic_energy(state, mass) - core.potential_energy(state, stiffness)
    return thermal_capacity * (internal / (thermal_capacity * reference_temperature)).log()


def rates_of(state: core.State) -> tuple[core.State, core.State]:
    return core.rates(state, mass, stiffness, friction, thermal_capacity, reference_temperature)


def release(n_frames: int, steps_per_frame: int, dt: float) -> Iterator[core.State]:
    """The state at each frame, so many steps of the combined motion apart."""
    state = start
    for _ in range(n_frames):
        yield state
        for _ in range(steps_per_frame):
            state = core.advance(state, mass, stiffness, friction, thermal_capacity, reference_temperature, dt)


if __name__ == "__main__":
    from examples.util.animation import save_animation, save_figure
    from examples.thermodynamics.generic import render

    n_frames, steps_per_frame, dt, frame_duration_ms = 150, 2, 0.04, 40
    save_figure(render.draw(energies_of, surface_entropy_of, rates_of, start), "generic_thermodynamics")
    save_animation(render.damping(energies_of, surface_entropy_of, rates_of, release(n_frames, steps_per_frame, dt),
                                  steps_per_frame * dt), "generic_thermodynamics", frame_duration_ms)
