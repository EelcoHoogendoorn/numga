"""Scenario parameters sweep the Hubbard ring toward its Heisenberg spectrum and evolve a flipped spin around the ring."""

from __future__ import annotations

import numpy as np

from examples.quantum.hubbard_ring import core
from numga import stack

hopping_strength = 1.0
n_samples = 157
n_spin_levels = 16
repulsion_sweep = hopping_strength * np.linspace(1.0, 40.0, n_samples)
heisenberg_levels = np.array([1.0, 2.0, 3.0])

n_cases = 3
repulsions = hopping_strength * np.array([4.0, 8.0, 20.0])
n_frames = 189
exchange_phases = np.linspace(0.0, 2 * np.pi, n_frames)
frame_duration_ms = 40

# Spin up on the first three sites and spin down on the fourth.
initial_state = core.mv.a ^ core.mv.c ^ core.mv.e ^ core.mv.h      # [] State+


def main() -> None:
    """The scenario entry point writes the excitation-level figure and flipped-spin walk animation and checks their limits."""
    from examples.quantum.hubbard_ring import render
    from examples.util.animation import save_animation, save_figure

    excitations = core.spin_excitations(hopping_strength, repulsion_sweep, n_spin_levels)   # [n_samples, n_spin_levels] Scalar+
    save_figure(render.levels_figure(repulsion_sweep / hopping_strength, excitations, heisenberg_levels),
                "hubbard_ring_levels")

    recorded = []
    save_animation(
        render.ring_frames(
            repulsions / hopping_strength,
            (recorded.append(frame) or frame
             for frame in core.spin_walk(hopping_strength, repulsions, initial_state, exchange_phases)),
        ),
        "hubbard_ring_walk",
        frame_duration_ms,
    )
    occupations = stack(recorded)                                   # [n_frames, n_cases, 4, 2] Scalar+

    # checks
    # Total electron count stays 4 across every frame and repulsion; at strong repulsion the spin levels
    # approach the Heisenberg ring's spectrum, and the flipped spin reaches the opposite site halfway through.
    assert ((occupations.sum(axis=(-1, -2)) - 4).abs() < 1e-9).all()
    assert ((excitations[-1, 1:4] - heisenberg_levels[0]).abs() < 0.05).all()
    assert ((excitations[-1, 4:11] - heisenberg_levels[1]).abs() < 0.05).all()
    assert ((excitations[-1, 11:16] - heisenberg_levels[2]).abs() < 0.05).all()
    assert occupations[n_frames // 2, -1, 1, 1] > 0.95


if __name__ == "__main__":
    main()
