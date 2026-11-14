"""Coverage test stepping the hydromagnetic dynamo once on a coarse ball and exercising its figures and animations."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from numga import stack
from examples.electromagnetism.dynamo import core, render, scenarios

mv = core.mv


def test_dynamo_steps_and_renders() -> None:
    n_scalar_modes, n_modes, n_steps = 4, 2, 1
    topology, vertices = scenarios.ball_complex(n_levels=1, n_shells=3)
    centres = topology.incidence[3, 0].mean(axis=0) * vertices                  # Point-[T]
    volumes, whitney, curl = core.whitney_operators(topology, vertices)
    stiffness = core.magnetic_stiffness(volumes, curl)                          # (Antiscalar- <- Scalar+)[E <- E]
    fluid_topology, fluid_vertices, embedding = scenarios.outer_core(topology, vertices)
    decay_rates, mode_velocities, mode_vorticities = core.flow_modes(
        fluid_topology, fluid_vertices, embedding, n_scalar_modes, n_modes
    )
    target_momentum = volumes * scenarios.helical_flow(topology, vertices).dual()  # Plane+[T]
    target_amplitudes = (target_momentum & mode_velocities).sites.sum()         # [n_modes] Scalar+
    radii = centres.antinorm()                                                  # Scalar+[T]
    conductivity = mv.scalar(
        np.where(radii < 1, scenarios.reynolds, scenarios.reynolds * scenarios.mantle_conductance)[:, None]
    ).field()                                                                   # Scalar+[T]
    initial_potential = scenarios.seed_potential(topology, vertices, scenarios.seed_field)  # Scalar+[E]
    states = list(
        core.dynamo_states(
            volumes, whitney, curl, decay_rates, mode_velocities, mode_vorticities,
            conductivity, scenarios.viscosity, scenarios.pull, target_amplitudes, initial_potential,
            scenarios.dt, n_steps,
        )
    )
    amplitudes, potentials = map(stack, zip(*states))                           # [n_steps + 1, n_modes] Scalar+, [n_steps + 1] Scalar+[E]
    kinetic_energy = (amplitudes * amplitudes).sum(axis=-1) / 2                 # [n_steps + 1] Scalar+
    magnetic_energy = (potentials & stiffness(potentials)).sites.sum() / 2      # [n_steps + 1] Scalar+

    final_velocity = (amplitudes[-1] * mode_velocities).sum(axis=0)             # Point-[T]
    volume_energy = (volumes * (final_velocity.dual() & final_velocity)).sites.sum() / 2  # Scalar+
    assert (volume_energy - kinetic_energy[-1]).abs() < 1e-10 * kinetic_energy[-1]
    target_velocity = (target_amplitudes * mode_velocities).sum(axis=0)         # Point-[T]
    _, advection_rates = core.flow_advection(
        volumes, curl, mode_velocities, mode_vorticities, target_velocity, amplitudes[-1], potentials[-1]
    )
    power_scale = ((advection_rates * advection_rates).sum() * (amplitudes[-1] * amplitudes[-1]).sum()).square_root()
    assert (amplitudes[-1] * advection_rates).sum().abs() < 1e-10 * power_scale
    face_fluxes = topology.d[1] * potentials[-1]                                # Scalar+[F]
    assert ((topology.d[2] * face_fluxes).norm().batch() < 1e-10 * face_fluxes.norm().batch().max()).all()

    single_frame = list(scenarios.field_frames(curl, mode_velocities, states[:1]))
    assert next(render.lines_animation(centres, single_frame)).ndim == 3
    assert next(render.cutaway_animation(centres, scenarios.mantle_radius, single_frame)).ndim == 3
    plt.close(render.modes_figure(centres, scenarios.mantle_radius, mode_velocities[:1]))
    plt.close(render.energies_figure(np.arange(n_steps + 1) * scenarios.dt, kinetic_energy, magnetic_energy))
