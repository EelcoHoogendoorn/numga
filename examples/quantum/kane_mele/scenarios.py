"""Scenes for the flake: its levels as the mass grows past the spin-orbit coupling, the state nearest
zero energy on either side, and an electron spreading from the edge, its two spins running around
the flake in opposite senses.

Energies are in units of the hop between neighbours, lengths in lattice constants.
"""

from __future__ import annotations

import numpy as np

from numga import stack
from examples.quantum.kane_mele import core

mv = core.mv
# The flake's size, its edge to centre in half lattice constants, and the outer share of it counted
# as its rim.
SIZE = 12
RIM = 0.3
# The spin-orbit coupling, and the masses swept past it.
SPIN_ORBIT = 0.1
MASSES = np.linspace(0.0, 0.8, 41)
# The levels nearest zero followed through the sweep, among the spin-up states, which hold each
# energy twice.
LEVELS = 32
# The masses whose states nearest zero are drawn: below the spin-orbit coupling's 3 * 3**0.5 times,
# and above.
CASES = np.array([0.0, 0.8])
# The states inside the gap without a mass, each energy four times, and the electron's spinor, spin
# along x, half up and half down.
IN_GAP = 32
SPINOR = (1 - mv.zx) * 0.5**0.5
# The time for the edge states to run once around the flake, and the frames over it.
LAP = 50.0
FRAMES = 50


# --- math -----------------------------------------------------------------------------
def sweep(flake: core.Flake, masses: np.ndarray):
    """The spin-up levels nearest zero for each mass, and how much of each lies on the rim."""
    spectra = [core.levels(core.hamiltonian(flake, SPIN_ORBIT, mass), core.Up, LEVELS) for mass in masses]
    energies = stack([energy for energy, _ in spectra])                         # [masses, levels] Scalar
    shares = stack([core.rim_share(flake, states) for _, states in spectra])    # [masses, levels] Scalar

    # --- checks
    # The Hamiltonian is its own reverse: every coupling's reverse couples the other way.
    energy = core.hamiltonian(flake, SPIN_ORBIT, masses[0])
    np.testing.assert_array_equal((~energy).cells.kernel, energy.cells.kernel)
    np.testing.assert_array_equal((~energy).columns, energy.columns)
    # Every spin-up level holds its energy twice, the state and the state times mv.xy.
    ordered = np.sort(energies.to_array(), axis=-1)
    np.testing.assert_allclose(ordered[:, ::2], ordered[:, 1::2], atol=1e-8)
    # Without a mass the levels inside the bulk gap, half of it 3 * 3**0.5 * SPIN_ORBIT, lie on the
    # rim; with the mass well past the spin-orbit coupling, the gap is empty.
    half = np.abs(3 * 3**0.5 * SPIN_ORBIT - masses)
    inside = np.abs(energies.to_array()[0]) < 0.9 * half[0]
    assert inside.sum() >= 8 and np.all(shares.to_array()[0, inside] > 0.9)
    assert np.abs(energies.to_array()[-1]).min() > 0.5 * half[-1]
    return energies, shares


def nearest(flake: core.Flake, masses: np.ndarray):
    """The density of the spin-up state nearest zero for each mass."""
    states = [core.levels(core.hamiltonian(flake, SPIN_ORBIT, mass), core.Up, 2)[1][0] for mass in masses]
    densities = stack(states).scalar_norm_squared()                            # [masses, atoms] Scalar

    # --- checks
    # Without a mass the state nearest zero lies on the rim; with a large one it spreads inside.
    shares = ((densities * flake.rim).sum(axis=-1) / densities.sum(axis=-1)).to_array()
    assert shares[0] > 0.9 and shares[-1] < 0.6
    return densities


def helical(flake: core.Flake):
    """An electron started at the middle of the edge facing x, spin along x, keeping its part inside
    the gap without a mass: its spin density at each frame over one lap."""
    energies, states = core.levels(core.hamiltonian(flake, SPIN_ORBIT, 0.0), core.Even, IN_GAP)   # [states] Scalar, [states, atoms] Even
    times = np.linspace(0.0, LAP, FRAMES + 1)
    spins = stack(list(core.spread(states, energies, flake.start, SPINOR, times)))   # [frames + 1, atoms] Vector

    # --- checks
    # Every state is inside the bulk gap and on the rim, and every energy is held four times: the two
    # spins, each with its state times mv.xy.
    ordered = np.sort(energies.to_array())
    assert np.abs(ordered).max() < 3 * 3**0.5 * SPIN_ORBIT
    np.testing.assert_allclose(ordered[::4], ordered[3::4], atol=1e-8)
    assert core.rim_share(flake, states).to_array().min() > 0.9
    # The density keeps its total, and the electron stays on the rim.
    density = spins.norm()                                                     # [frames + 1, atoms] Scalar
    totals = density.sum(axis=-1).to_array()
    np.testing.assert_allclose(totals, totals[0], rtol=1e-9)
    np.testing.assert_array_less(0.9 * totals, (density * flake.rim).sum(axis=-1).to_array())
    # A quarter lap on, the spin-up half has turned clockwise around the flake and the spin-down half
    # counterclockwise: the sine of each half's turn from the start, by its centre.
    along = spins[FRAMES // 4] | mv.z                                          # [atoms] Scalar
    up, down = (density[FRAMES // 4] + along) * 0.5, (density[FRAMES // 4] - along) * 0.5
    start = flake.positions[flake.start].normalized()                         # [] Vector
    centres = [((half * flake.positions).sum(axis=0) / half.sum(axis=0)).normalized() for half in (up, down)]
    sines = [(mv.xy | (centre ^ start)).to_array().item() for centre in centres]
    assert sines[0] < -0.5 and sines[1] > 0.5
    return spins


if __name__ == "__main__":
    from examples.animation import save_animation, save_figure
    from examples.quantum.kane_mele import render

    flake = core.flake(SIZE, RIM)
    save_figure(render.draw_levels(MASSES, *sweep(flake, MASSES), 3 * 3**0.5 * SPIN_ORBIT), "kane_mele_levels")
    save_figure(render.draw_states(flake, nearest(flake, CASES), [f"mass {mass}" for mass in CASES]), "kane_mele_states")
    save_animation(render.animate_spin(flake, helical(flake)), "kane_mele_helical", 120)
