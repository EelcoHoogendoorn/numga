"""Small-flake coverage test for the Kane–Mele topological insulator: Hamiltonian Hermiticity,
mass-sweep spectrum, gap-edge densities, helical spin evolution, and all four renderers."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.quantum.kane_mele import core, render, scenarios


def test_small_flake_spectrum_evolution_and_figures():
    flake = scenarios.honeycomb_flake(6, scenarios.rim_fraction)
    energy = core.hamiltonian(
        flake.bonds, flake.next_bonds, flake.positions, flake.between,
        flake.sublattice, scenarios.spin_orbit, 0.0,
    )
    in_gap, edge_states = core.sector_levels(energy, core.Even, 8)
    assert (((~energy - energy) * edge_states).scalar_norm_squared().sites.sum() < 1e-20).all()

    energies, shares = scenarios.mass_sweep(flake, scenarios.sample_masses, 4)
    densities = scenarios.gap_edge_densities(flake, scenarios.sample_masses)
    spins = core.spin_evolution(
        edge_states, in_gap, flake.start_site, scenarios.initial_spinor, np.array([0.0, 1.0]),
    )

    plt.close(render.flake_figure(flake))
    plt.close(render.levels_figure(scenarios.sample_masses, energies, shares, scenarios.gap_threshold))
    plt.close(render.states_figure(flake, densities))
    frame = next(render.spin_animation(flake, 64, spins))
    assert frame.ndim == 3 and frame.shape[-1] == 3
    plt.close("all")
