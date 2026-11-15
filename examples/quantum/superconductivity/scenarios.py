"""Pairing-strength quench scenarios and collective linear response of the superconducting Higgs mode."""

from __future__ import annotations

from collections.abc import Iterator
from functools import partial

import numpy as np

from numga import stack
from examples.quantum.superconductivity import core

n_levels = 240
cutoff = 5.0
coupling = 4.4
quench_ratio = 0.88
small_quench_ratio = 0.99
n_alignment_iterations = 160
dt = 0.04
n_steps = 600
n_steps_per_frame = 2
n_midpoint_iterations = 3
frame_duration_ms = 40
feedback_strengths = np.array([1.0, 0.0])
response_names = ("collective response", "fixed pairing field")


# --- math -----------------------------------------------------------------------------
def quench_states(
    dispersion: core.Field, initial_spins: core.Spin, post_pairing: core.Pairing, n_run_steps: int,
) -> Iterator[tuple[core.Spin, core.Field]]:
    """Yields `(spins, gap)` frames across `n_run_steps` implicit midpoint steps after a pairing quench."""
    rate = partial(core.precession_rate, dispersion=dispersion, pairing=post_pairing)
    for spins in core.evolution(initial_spins, rate, dt, n_run_steps, n_steps_per_frame, n_midpoint_iterations):
        yield spins, post_pairing(spins.sites.mean())


def linear_response(
    dispersion: core.Field, pairing: core.Pairing, equilibrium: core.Spin,
    disturbance: core.Spin, n_run_steps: int,
) -> core.Spin:
    """Evolves `disturbance` with and without collective gap feedback (`[n_frames, 2] Spin-[levels]`)."""
    local_map, feedback_map = core.response_maps(dispersion, pairing, equilibrium)
    feedback_maps = feedback_map * feedback_strengths                         # [2] (Spin- <- Spin-)[levels]
    batched_disturbance = disturbance.broadcast_to(feedback_strengths.shape)  # [2] Spin-[levels]
    return stack(core.evolution(
        batched_disturbance,
        lambda delta: local_map(delta) + feedback_maps(delta.sites.mean()),
        dt, n_run_steps, n_steps_per_frame, n_midpoint_iterations,
    ))


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    """Generates the equilibrium, quench animation, gap oscillation, and linear response figures."""
    from examples.util.animation import save_animation, save_figure
    from examples.quantum.superconductivity import render

    dispersion = core.energy_levels(n_levels, cutoff)                         # Field+[levels]
    pairing = coupling * core.coherence                                       # [] Field+ <- Spin-
    seed_gap = core.mv.yz                                                     # [] Field+
    initial_spins = core.ground_state(dispersion, pairing, seed_gap, n_alignment_iterations)  # Spin-[levels]
    save_figure(render.equilibrium_figure(dispersion, initial_spins), "superconductivity_equilibrium")

    post_pairing = quench_ratio * pairing                                     # [] Field+ <- Spin-
    n_frames = n_steps // n_steps_per_frame + 1
    times = np.arange(n_frames) * n_steps_per_frame * dt
    recorded = []

    def stream() -> Iterator[tuple[core.Spin, core.Field]]:
        for state in quench_states(dispersion, initial_spins, post_pairing, n_steps):
            recorded.append(state)
            yield state

    save_animation(render.quench_animation(dispersion, times, stream()), "superconductivity_quench", frame_duration_ms)

    quench_history = stack([spins for spins, _ in recorded])                  # [n_frames] Spin-[levels]
    gaps = stack([gap for _, gap in recorded])                                # [n_frames] Field+
    final_equilibrium = core.ground_state(dispersion, post_pairing, seed_gap, n_alignment_iterations)  # Spin-[levels]
    final_gap = post_pairing(final_equilibrium.sites.mean())                  # [] Field+
    save_figure(render.gap_figure(times, gaps, final_gap), "superconductivity_gap")

    weak_pairing = small_quench_ratio * pairing                               # [] Field+ <- Spin-
    weak_equilibrium = core.ground_state(dispersion, weak_pairing, seed_gap, n_alignment_iterations)  # Spin-[levels]
    weak_gap = weak_pairing(weak_equilibrium.sites.mean())                    # [] Field+
    weak_history = stack([spins for spins, _ in quench_states(dispersion, initial_spins, weak_pairing, n_steps)])
    disturbance_history = linear_response(
        dispersion, weak_pairing, weak_equilibrium, initial_spins - weak_equilibrium, n_steps,
    )
    exact_gap_change = weak_pairing(weak_history.sites.mean()) - weak_gap
    predicted_gap_change = weak_pairing(disturbance_history.sites.mean())
    save_figure(
        render.response_figure(times, exact_gap_change, predicted_gap_change, response_names),
        "superconductivity_response",
    )

    # checks
    initial_gap = pairing(initial_spins.sites.mean())
    assert ((initial_spins + (dispersion - initial_gap).dual().normalized() / 2).antinorm() < 1e-12).all()
    energies = core.bcs_energy(quench_history, dispersion, post_pairing)
    assert ((energies - energies[0]).abs() < 1e-4).all()
    assert core.bcs_energy(final_equilibrium, dispersion, post_pairing) < energies[0]
    occupations = (core.mv.xy & quench_history).sites.mean()
    assert ((occupations - occupations[0]).abs() < 1e-12).all()
    assert ((quench_history.antinorm() - 0.5).abs() < 1e-3).all()
    assert ((exact_gap_change - predicted_gap_change[:, 0]).norm() < 1e-4).all()
    assert ((exact_gap_change - predicted_gap_change[:, 1]).norm() > 5e-3).any()


if __name__ == "__main__":
    main()
