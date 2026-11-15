"""A displaced mass released on a damped spring hands its mechanical energy to a thermal reservoir."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import stack
from examples.thermodynamics.generic import core

mass, stiffness, friction = 1.0, 4.0, 0.35
thermal_capacity, reference_temperature = 2.0, 1.0
displacement = 2.0

# The mass starts displaced at rest while the reservoir starts at the reference temperature with zero entropy.
initial_state = core.mv.q.dual() * displacement + core.mv.p.dual() * 0.0 + core.mv.s.dual() * 0.0   # [] State-
initial_energy = core.total_energy(
    initial_state, mass, stiffness, thermal_capacity, reference_temperature
)                                                                                                   # [] Scalar+


# --- math -----------------------------------------------------------------------------
def energies_of(state: core.State) -> tuple[core.Scalar, core.Scalar, core.Scalar]:
    """Kinetic, elastic and reservoir internal energy at each state."""
    return (
        core.kinetic_energy(state, mass),
        core.potential_energy(state, stiffness),
        core.internal_energy(state, thermal_capacity, reference_temperature),
    )


def surface_entropy_of(state: core.State) -> core.Scalar:
    """Reservoir entropy as a function of position and momentum when total energy equals `initial_energy`."""
    internal = initial_energy - core.kinetic_energy(state, mass) - core.potential_energy(state, stiffness)
    return thermal_capacity * (internal / (thermal_capacity * reference_temperature)).log()


def release(n_frames: int, n_steps_per_frame: int, dt: float) -> Iterator[core.State]:
    """Yield the oscillator state at each frame, advancing `n_steps_per_frame` Runge-Kutta steps between frames."""
    state = initial_state
    for _ in range(n_frames):
        yield state
        for _ in range(n_steps_per_frame):
            state = core.advance(state, mass, stiffness, friction, thermal_capacity, reference_temperature, dt)


def main() -> None:
    from examples.thermodynamics.generic import render
    from examples.util.animation import save_animation, save_figure

    n_frames, n_steps_per_frame, dt, frame_duration_ms = 150, 2, 0.04, 40
    trajectory = list(release(n_frames, n_steps_per_frame, dt))
    save_figure(render.draw(energies_of, surface_entropy_of, trajectory[0]), "generic_thermodynamics")
    save_animation(
        render.damping(energies_of, surface_entropy_of, n_steps_per_frame * dt, iter(trajectory)),
        "generic_thermodynamics",
        frame_duration_ms,
    )

    # checks
    states = stack(trajectory)                                                                      # [n_frames] State-
    kinetic, elastic, internal = energies_of(states)                                                # [n_frames] Scalar+, Scalar+, Scalar+
    entropies = core.mv.s & states                                                                  # [n_frames] Scalar+
    np.testing.assert_allclose(((kinetic + elastic + internal) - initial_energy).kernel, 0, atol=1e-4)
    assert np.all(entropies[1:] - entropies[:-1] >= -1e-8)
    np.testing.assert_allclose(core.poisson_map(core.entropy_potential).kernel, 0, atol=1e-6)
    sample_mobility = core.friction_mobility(trajectory[1], mass, friction, thermal_capacity, reference_temperature)
    sample_potential = core.energy_potential(trajectory[1], mass, stiffness, thermal_capacity, reference_temperature)
    np.testing.assert_allclose(sample_mobility(sample_potential).kernel, 0, atol=1e-6)


if __name__ == "__main__":
    main()
