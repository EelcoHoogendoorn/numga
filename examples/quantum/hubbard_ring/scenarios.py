"""The Hubbard ring: its low levels approaching the Heisenberg ring, and a flipped spin walking around."""

from __future__ import annotations

import numpy as np

from examples.quantum.hubbard_ring import core

HOPPING = 1.0
# The levels: a sweep of the repulsion, the sixteen spin states that one electron per site allows, and
# where the Heisenberg ring of four spins puts them above its ground state, in units of the exchange.
SWEEP = np.linspace(1.0, 40.0, 157)
SPIN_STATES = 16
HEISENBERG = np.array([1.0, 2.0, 3.0])
# The walk: repulsions, one cycle of the exchange phase, and each frame's duration.
REPULSIONS = HOPPING * np.array([4.0, 8.0, 20.0])
FRAMES = 151
PHASE = np.linspace(0.0, 2 * np.pi, FRAMES)
FRAME_MS = 50
# Spin up on the first three sites, spin down on the last.
START = core.mv.a ^ core.mv.c ^ core.mv.e ^ core.mv.h


# --- math -----------------------------------------------------------------------------
def exchange(repulsion: np.ndarray) -> np.ndarray:
    """The exchange between neighbouring spins at strong repulsion: four times hopping squared over it."""
    return 4 * HOPPING**2 / repulsion


def levels(repulsion: np.ndarray) -> core.Scalar:
    """The lowest levels above the ground state, in units of the exchange."""
    energies = core.hamiltonian(HOPPING, repulsion).eigvalsh()  # [samples, modes] Scalar
    return (energies[:, :SPIN_STATES] - energies[:, :1]) / exchange(repulsion)[:, None]   # [samples, spin states] Scalar


def walk(repulsion: np.ndarray, phase: np.ndarray) -> core.Scalar:
    """The flipped spin over one cycle of the exchange phase: how many electrons each orbital holds."""
    energies, modes = core.hamiltonian(HOPPING, repulsion).eigh()   # [cases, modes] Scalar, State
    times = phase / exchange(repulsion)[:, None]               # [cases, times]
    return core.occupation(core.turn(energies, modes, START, times))   # [cases, times, sites, spins] Scalar


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.animation import save_animation, save_figure
    from examples.quantum.hubbard_ring import render

    excitations = levels(SWEEP)
    occupation = walk(REPULSIONS, PHASE)
    save_figure(render.draw_levels(SWEEP / HOPPING, excitations, HEISENBERG), "hubbard_ring_levels")
    save_animation(render.animate_ring([f"$U/t = {ratio:g}$" for ratio in REPULSIONS / HOPPING], occupation),
                   "hubbard_ring_walk", FRAME_MS)

    # --- checks
    # Four electrons all along; at strong repulsion the spin levels approach the Heisenberg ring's, and
    # the flipped spin reaches the opposite site halfway through the cycle.
    counts = occupation.to_array()                              # [cases, times, sites, spins]
    np.testing.assert_allclose(counts.sum(axis=(-1, -2)), 4.0, atol=1e-11)
    np.testing.assert_allclose(excitations.to_array()[-1, 1:4], HEISENBERG[0], atol=0.05)
    assert counts[-1, FRAMES // 2, 1, 1] > 0.95
    print("Hubbard ring checks passed.")


if __name__ == "__main__":
    main()
