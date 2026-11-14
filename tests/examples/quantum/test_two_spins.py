"""Coverage tests exercise two-spin exchange, steering, magnetic field levels, singlet-triplet precession, and rendering."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.quantum.two_spins import core, render, scenarios


def test_the_scenes_draw_at_a_few_samples() -> None:
    """Minimal sample counts verify two-spin math invariants and render every figure and animation frame."""
    singlet_state, singlet_bell = scenarios.singlet_bound()
    assert ((2 * singlet_state.scalar_norm_squared() - 1).abs() < 1e-12).all()
    assert ((singlet_bell - 2 * np.sqrt(2)).abs() < 1e-12).all()

    angles, first_planes, second_planes, bells, states = scenarios.exchange_swap(3)
    assert ((bells[1] - 2 * np.sqrt(2)).abs() < 1e-9).all()
    plt.close(render.draw_state(first_planes[0], second_planes[0], render.plane_reading))
    plt.close(render.draw_exchange(angles, bells, first_planes, render.plane_reading))
    plt.close(render.draw_entanglement(angles, core.joint_map(states).svdvals(), first_planes, render.plane_reading))

    planes, probability, steered_planes = scenarios.steering_grid(states[:1], 2, 2)
    assert ((probability - 0.5 * (1 + planes.dot(first_planes[:1, None, None]))).abs() < 1e-11).all()
    frames = render.animate_steering(planes, probability, steered_planes, render.plane_reading)
    frame = next(frames)
    assert frame.ndim == 3
    frames.close()

    fields = np.linspace(0.0, scenarios.max_field, 2)
    basis_first, basis_second = core.spin_planes(scenarios.basis_states)
    energies = scenarios.field_sweep(fields, scenarios.exchange_rate)
    plt.close(render.draw_levels(fields, energies, core.field_plane.dot(basis_first + basis_second)))

    times, rates = np.linspace(0.0, np.pi, 2), scenarios.exchange_rates[:1]
    qubit_states, singlet_return = scenarios.singlet_oscillation(times, rates, scenarios.difference)
    assert ((2 * qubit_states.scalar_norm_squared() - 1).abs() < 1e-12).all()
    plt.close(render.draw_qubit(*core.qubit_coordinates(qubit_states), times, singlet_return, rates))
