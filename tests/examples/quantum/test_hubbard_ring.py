"""Coverage tests verify the four-electron derivation lift, double occupancy, Heisenberg limit, spin walk, and rendering."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.quantum.hubbard_ring import core, render, scenarios


def test_derivation_lift_and_double_occupancy():
    mv = core.mv
    first, second, third, fourth = mv.a, mv.d, mv.e, mv.h
    state = first ^ second ^ third ^ fourth                         # [] State+
    derivation_sum = ((core.orbital_transfer(first) ^ second ^ third ^ fourth)
                      + (first ^ core.orbital_transfer(second) ^ third ^ fourth)
                      + (first ^ second ^ core.orbital_transfer(third) ^ fourth)
                      + (first ^ second ^ third ^ core.orbital_transfer(fourth)))   # [] State+
    assert (core.ring_hopping(state) - derivation_sum).scalar_norm_squared() < 1e-9

    paired_state = mv.a ^ mv.b ^ mv.c ^ mv.d                        # [] State+
    assert (core.double_occupancy(paired_state) - 2 * paired_state).scalar_norm_squared() < 1e-9


def test_heisenberg_limit_walk_and_rendering():
    repulsions = np.array([10.0, 40.0])
    excitations = core.spin_excitations(1.0, repulsions, scenarios.n_spin_levels)   # [2, 16] Scalar+
    assert ((excitations[-1, 1:4] - 1).abs() < 0.05).all()
    assert ((excitations[-1, 4:11] - 2).abs() < 0.05).all()
    assert ((excitations[-1, 11:16] - 3).abs() < 0.05).all()

    figure = render.levels_figure(repulsions, excitations, scenarios.heisenberg_levels)
    plt.close(figure)

    walk_repulsions = np.array([40.0])
    phases = np.array([0.0, np.pi])
    occupations = list(core.spin_walk(1.0, walk_repulsions, scenarios.initial_state, phases))
    assert len(occupations) == 2
    assert ((occupations[0].sum(axis=(-1, -2)) - 4).abs() < 1e-9).all()
    assert occupations[1][0, 1, 1] > 0.95

    frames = list(render.ring_frames(walk_repulsions, occupations))
    assert len(frames) == 2
    assert frames[0].ndim == 3
