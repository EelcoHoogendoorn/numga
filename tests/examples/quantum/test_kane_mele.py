"""The Kane–Mele flake: its spin-up levels are those of the model's complex Hamiltonian, and the scenes
pass their checks and draw."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.quantum.kane_mele import core, render, scenarios


def test_spin_up_levels_are_those_of_the_complex_hamiltonian():
    """With the spin-orbit hop i times the turn's sense on each pair of second neighbours, the
    complex Hamiltonian of the spin-up amplitudes has the levels of the spin-up spinors."""
    flake = core.flake(8, scenarios.RIM)
    spin_orbit, mass, count = 0.1, 0.3, 12
    complex_form = np.diag(mass * flake.sublattice).astype(complex)
    complex_form[tuple(flake.first.T)] = -1.0
    complex_form[tuple(flake.second.T)] += 1j * spin_orbit * flake.senses
    expected = np.linalg.eigvalsh(complex_form)
    energies, _ = core.levels(core.hamiltonian(flake, spin_orbit, mass), core.Up, 2 * count)
    nearest = np.sort(expected[np.argsort(np.abs(expected))[:count]])
    np.testing.assert_allclose(np.sort(energies.to_array())[::2], nearest, atol=1e-10)


def test_scenes_pass_their_checks_and_draw():
    flake = core.flake(scenarios.SIZE, scenarios.RIM)
    spins = scenarios.helical(flake)
    for figure in (render.draw_levels(scenarios.MASSES, *scenarios.sweep(flake, scenarios.MASSES), 3 * 3**0.5 * scenarios.SPIN_ORBIT),
                   render.draw_states(flake, scenarios.nearest(flake, scenarios.CASES), ["", ""])):
        assert isinstance(figure, plt.Figure)
        plt.close(figure)
    assert len(render.animate_spin(flake, spins[:2])) == 2
