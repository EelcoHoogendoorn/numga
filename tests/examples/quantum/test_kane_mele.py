"""The Kane–Mele flake: its spin-up levels are those of the model's complex Hamiltonian, the edge states lie on
the rim with the two spins circling oppositely, and the scenes draw."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.quantum.kane_mele import core, render, scenarios


def test_spin_up_levels_are_those_of_the_complex_hamiltonian():
    """With the spin-orbit hop i times the sense of the turn on each pair of second neighbours, the
    sign of the patch its path's two steps span, read by the xy plane, the complex Hamiltonian of the
    spin-up amplitudes has the levels of the spin-up spinors."""
    flake = core.flake(8, scenarios.RIM)
    spin_orbit, mass, n_levels = 0.1, 0.3, 12
    complex_form = np.diag(mass * flake.sublattice).astype(complex)
    complex_form[tuple(flake.first.T)] = -1.0
    near = flake.positions.batch()[flake.second[:, 0]]
    far = flake.positions.batch()[flake.second[:, 1]]
    senses = np.sign((core.mv.xy & ((flake.between - far) & (near - flake.between))).to_array())
    complex_form[tuple(flake.second.T)] += 1j * spin_orbit * senses
    expected = np.linalg.eigvalsh(complex_form)
    energies, _ = core.levels(core.hamiltonian(flake, spin_orbit, mass), core.Up, 2 * n_levels)
    nearest = np.sort(expected[np.argsort(np.abs(expected))[:n_levels]])
    np.testing.assert_allclose(np.sort(energies.to_array())[::2], nearest, atol=1e-10)


def test_edge_states_lie_on_the_rim_and_the_spins_circle_oppositely():
    flake = core.flake(scenarios.SIZE, scenarios.RIM)
    energies, states, spins = scenarios.helical(flake)
    assert core.rim_share(flake, states).to_array().min() > 0.9
    # A quarter lap on, the spin-up half has turned counterclockwise around the flake and the spin-down half
    # clockwise: the sine of each half's turn from the start, by its centre.
    quarter = scenarios.N_FRAMES // 4
    density, along = spins[quarter].norm(), core.mv.xy & spins[quarter]
    start = flake.positions.batch()[flake.start].normalized()
    sines = []
    for half in ((density + along) * 0.5, (density - along) * 0.5):
        centre = ((half * flake.positions).sites.sum() / half.sites.sum()).normalized()
        sines.append((core.mv.xy & (start & centre)).to_array().item())
    assert sines[0] > 0.5 and sines[1] < -0.5


def test_scenes_draw():
    flake = core.flake(scenarios.SIZE, scenarios.RIM)
    _, _, spins = scenarios.helical(flake)
    for figure in (render.draw_levels(scenarios.MASSES, *scenarios.sweep(flake, scenarios.MASSES), 3 * 3**0.5 * scenarios.SPIN_ORBIT),
                   render.draw_states(flake, scenarios.nearest(flake, scenarios.CASES))):
        assert isinstance(figure, plt.Figure)
        plt.close(figure)
    assert len(list(render.animate_spin(flake, spins[:2], render.PIXELS))) == 2
