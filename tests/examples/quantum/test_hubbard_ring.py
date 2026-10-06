"""Four electrons on a ring: the lift's product rule at four electrons, the Heisenberg limit, and a
flipped spin crossing the ring."""

from __future__ import annotations

import numpy as np

from examples.quantum.hubbard_ring import core, scenarios


def test_the_lift_acts_on_each_of_four_electrons_in_turn():
    mv = core.mv
    single = -(core.FOLLOWING * (core.ORBITALS | core.Orbital) + core.ORBITALS * (core.FOLLOWING | core.Orbital)).sum(axis=0).sum(axis=0)
    first, second, third, fourth = mv.a, mv.d, mv.e, mv.h
    rule = ((single(first) ^ second ^ third ^ fourth) + (first ^ single(second) ^ third ^ fourth)
            + (first ^ second ^ single(third) ^ fourth) + (first ^ second ^ third ^ single(fourth)))
    np.testing.assert_allclose((core.hopping(1.0)(first ^ second ^ third ^ fourth) - rule).kernel, 0.0, atol=1e-12)
    # Two sites holding both spins, two holding none.
    np.testing.assert_allclose((core.double_occupancy()(mv.a ^ mv.b ^ mv.c ^ mv.d) - 2 * (mv.a ^ mv.b ^ mv.c ^ mv.d)).kernel, 0.0, atol=1e-12)


def test_spins_approach_the_heisenberg_ring_and_a_flipped_spin_crosses_it():
    excitations = scenarios.levels(np.array([40.0])).to_array()[0]
    np.testing.assert_allclose(excitations[1:4], 1.0, atol=0.05)
    np.testing.assert_allclose(excitations[4:11], 2.0, atol=0.05)
    np.testing.assert_allclose(excitations[11:16], 3.0, atol=0.05)
    phase = np.linspace(0.0, 2 * np.pi, 9)
    counts = scenarios.walk(np.array([40.0]), phase).to_array()[0]    # [times, sites, spins]
    np.testing.assert_allclose(counts.sum(axis=(-1, -2)), 4.0, atol=1e-11)
    # In the Heisenberg ring the flipped spin reaches the opposite site as the fourth power of the sine
    # of half the exchange phase; at finite repulsion, nearly.
    np.testing.assert_allclose(counts[:, 1, 1], np.sin(phase / 2) ** 4, atol=0.03)
