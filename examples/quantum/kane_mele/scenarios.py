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
N_LEVELS = 32
# The masses whose states nearest zero are drawn: below the spin-orbit coupling's 3 * 3**0.5 times,
# and above.
CASES = np.array([0.0, 0.8])
# The states inside the gap without a mass, each energy four times, and the electron's spinor, spin
# along x, half up and half down.
N_IN_GAP = 32
SPINOR = (1 - mv.zx) * 0.5**0.5
# The time for the edge states to run once around the flake, and the frames over it.
LAP = 50.0
N_FRAMES = 50
DURATION_MS = 120
# README and gallery picture: this makes plots/kane_mele.gif for the README and docs/extensors.md, tuned as a visual for both. Don't change it; to experiment, copy it.
GALLERY_PIXELS = 400


# --- math -----------------------------------------------------------------------------
def sweep(flake: core.Flake, masses: np.ndarray) -> tuple[core.Scalar, core.Scalar]:
    """The spin-up levels nearest zero for each mass, and how much of each lies on the rim."""
    spectra = [core.levels(core.hamiltonian(flake, SPIN_ORBIT, mass), core.Up, N_LEVELS) for mass in masses]
    energies = stack([energy for energy, _ in spectra])                        # [masses, levels] Scalar+
    shares = stack([core.rim_share(flake, states) for _, states in spectra])   # [masses, levels] Scalar+
    return energies, shares


def nearest(flake: core.Flake, masses: np.ndarray) -> core.Scalar:
    """The density of the spin-up state nearest zero for each mass."""
    states = [core.levels(core.hamiltonian(flake, SPIN_ORBIT, mass), core.Up, 2)[1][0] for mass in masses]
    densities = stack(states).scalar_norm_squared()                            # [masses] Scalar+[atoms]
    return densities


def helical(flake: core.Flake) -> tuple[core.Scalar, core.Even, core.Spin]:
    """The levels inside the gap without a mass and their states, and an electron started at the middle of the
    edge facing x, spin along x, keeping its part in those states: its spin density at each frame over one lap."""
    energies, states = core.levels(core.hamiltonian(flake, SPIN_ORBIT, 0.0), core.Even, N_IN_GAP)   # [states] Scalar+, [states] Even+[atoms]
    times = np.linspace(0.0, LAP, N_FRAMES + 1)
    spins = stack(list(core.spread(states, energies, flake.start, SPINOR, times)))   # [frames + 1] Spin-[atoms]
    return energies, states, spins


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.quantum.kane_mele import render

    flake = core.flake(SIZE, RIM)
    energies, shares = sweep(flake, MASSES)
    densities = nearest(flake, CASES)
    in_gap, states, spins = helical(flake)
    save_figure(render.draw_levels(MASSES, energies, shares, 3 * 3**0.5 * SPIN_ORBIT), "kane_mele_levels")
    save_figure(render.draw_states(flake, densities), "kane_mele_states")
    save_animation(render.animate_spin(flake, spins, render.PIXELS), "kane_mele_helical", DURATION_MS)
    # README and gallery picture: this makes plots/kane_mele.gif for the README and docs/extensors.md, tuned as a visual for both. Don't change it; to experiment, copy it.
    save_animation(render.animate_spin(flake, spins, GALLERY_PIXELS), "kane_mele", DURATION_MS)

    # checks
    # The Hamiltonian is its own reverse: every coupling's reverse couples the other way.
    energy = core.hamiltonian(flake, SPIN_ORBIT, MASSES[0])
    np.testing.assert_array_equal((~energy).cells.kernel, energy.cells.kernel)
    np.testing.assert_array_equal((~energy).columns, energy.columns)
    # Every spin-up level holds its energy twice, the state and the state times mv.xy.
    ordered = np.sort(energies.to_array(), axis=-1)
    np.testing.assert_allclose(ordered[:, ::2], ordered[:, 1::2], atol=1e-8)
    # Without a mass the levels inside the bulk gap, half of it 3 * 3**0.5 * SPIN_ORBIT, lie on the
    # rim; with the mass well past the spin-orbit coupling, the gap is empty.
    half = np.abs(3 * 3**0.5 * SPIN_ORBIT - MASSES)
    inside = np.abs(energies.to_array()[0]) < 0.9 * half[0]
    assert inside.sum() >= 8 and np.all(shares.to_array()[0, inside] > 0.9)
    assert np.abs(energies.to_array()[-1]).min() > 0.5 * half[-1]
    # Without a mass the state nearest zero lies on the rim; with a large one it spreads inside.
    nearest_shares = ((densities * flake.rim).sites.sum() / densities.sites.sum()).to_array()
    assert nearest_shares[0] > 0.9 and nearest_shares[-1] < 0.6
    # Every state is inside the bulk gap and on the rim, and every energy is held four times: the two
    # spins, each with its state times mv.xy.
    ordered = np.sort(in_gap.to_array())
    assert np.abs(ordered).max() < 3 * 3**0.5 * SPIN_ORBIT
    np.testing.assert_allclose(ordered[::4], ordered[3::4], atol=1e-8)
    assert core.rim_share(flake, states).to_array().min() > 0.9
    # The density keeps its total, and the electron stays on the rim.
    density = spins.norm()                                                     # [frames + 1] Scalar+[atoms]
    totals = density.sites.sum().to_array()
    np.testing.assert_allclose(totals, totals[0], rtol=1e-9)
    np.testing.assert_array_less(0.9 * totals, (density * flake.rim).sites.sum().to_array())
    # A quarter lap on, the spin-up half has turned counterclockwise around the flake and the spin-down
    # half clockwise: the sine of each half's turn from the start, by its centre.
    along = mv.xy & spins[N_FRAMES // 4]                                       # Scalar+[atoms]
    up, down = (density[N_FRAMES // 4] + along) * 0.5, (density[N_FRAMES // 4] - along) * 0.5
    start = flake.positions.batch()[flake.start].normalized()                  # [] Point-
    centres = [((half * flake.positions).sites.sum() / half.sites.sum()).normalized() for half in (up, down)]
    sines = [(mv.xy & (start & centre)).to_array().item() for centre in centres]
    assert sines[0] > 0.5 and sines[1] < -0.5


if __name__ == "__main__":
    main()
