"""Coverage tests for the superconducting condensate, its mirror covariance, and its figures."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from numga import stack
from examples.quantum.superconductivity import core, render, scenarios

tolerance = 1e-10


@pytest.fixture(scope="module")
def sample_quench():
    n_test_levels = 16
    n_test_alignment = 60
    n_test_steps = scenarios.n_steps_per_frame
    dispersion = core.energy_levels(n_test_levels, scenarios.cutoff)
    pairing = scenarios.coupling * core.coherence
    post_pairing = scenarios.quench_ratio * pairing
    initial_spins = core.ground_state(dispersion, pairing, core.mv.yz, n_test_alignment)
    equilibrium = core.ground_state(dispersion, post_pairing, core.mv.yz, n_test_alignment)
    states = list(scenarios.quench_states(dispersion, initial_spins, post_pairing, n_test_steps))
    times = np.arange(len(states)) * scenarios.n_steps_per_frame * scenarios.dt
    return dispersion, post_pairing, initial_spins, equilibrium, times, states


def test_invariants_and_collective_phase_zero_mode(sample_quench):
    dispersion, post_pairing, initial_spins, equilibrium, _, states = sample_quench
    history = stack([spins for spins, _ in states])
    energies = core.bcs_energy(history, dispersion, post_pairing)
    assert ((energies - energies[0]).abs() < 1e-5).all()
    occupations = (core.mv.xy & history).sites.mean()
    assert ((occupations - occupations[0]).abs() < tolerance).all()
    assert ((history.antinorm() - 0.5).abs() < 1e-5).all()

    # Turning every pairing phase together produces another equilibrium; collective feedback cancels local precession.
    local_map, feedback_map = core.response_maps(dispersion, post_pairing, equilibrium)
    phase_shift = -core.mv.xy.commutator(equilibrium)
    local_rate = local_map(phase_shift)
    assert (local_rate.antinorm() > 1e-2).any()
    assert ((local_rate + feedback_map(phase_shift.sites.mean())).antinorm() < tolerance).all()


def test_precession_and_ground_state_are_mirror_covariant(sample_quench):
    dispersion, post_pairing, initial_spins, _, _, _ = sample_quench
    rate = core.precession_rate(initial_spins, dispersion, post_pairing)
    for mirror in (core.mv.x, core.mv.z, core.mv.vector([3, 0, 4]).normalized()):
        mirrored_dispersion = mirror >> dispersion
        mirrored_pairing = mirror >> post_pairing(mirror << core.Spin)
        mirrored_rate = core.precession_rate(mirror >> initial_spins, mirrored_dispersion, mirrored_pairing)
        assert ((mirrored_rate - (mirror >> rate)).antinorm() < tolerance).all()
        mirrored_ground = core.ground_state(mirrored_dispersion, mirrored_pairing, mirror >> core.mv.yz, 10)
        expected_ground = mirror >> core.ground_state(dispersion, post_pairing, core.mv.yz, 10)
        assert ((mirrored_ground - expected_ground).antinorm() < tolerance).all()


def test_figures_and_single_frame_render(sample_quench):
    dispersion, post_pairing, initial_spins, equilibrium, times, states = sample_quench
    history = stack([spins for spins, _ in states])
    gaps = stack([gap for _, gap in states])
    equilibrium_gap = post_pairing(equilibrium.sites.mean())
    try:
        render.equilibrium_figure(dispersion, initial_spins)
        frames = render.quench_animation(dispersion, times, iter(states[:1]))
        frame = next(frames)
        frames.close()
        assert frame.ndim == 3 and frame.shape[2] == 3
        render.gap_figure(times, gaps, equilibrium_gap)
        disturbance = scenarios.linear_response(
            dispersion, post_pairing, equilibrium, initial_spins - equilibrium, scenarios.n_steps_per_frame,
        )
        render.response_figure(times, gaps - equilibrium_gap, post_pairing(disturbance.sites.mean()), scenarios.response_names)
    finally:
        plt.close("all")
