"""Scenes for the Dirac electron: the energies of plane waves over momentum, and the trembling path
of an electron with some negative energy mixed in.

Units are natural, with the reduced Planck constant and the speed of light both one, and the
electron's mass as the unit of energy: lengths are in units of the reduced Compton wavelength,
386 femtometres, and times in the time light takes to cross it.
"""

from __future__ import annotations

import numpy as np

from examples.relativity.dirac import core

mv = core.mv
MASS = 1.0
# The share of negative energy in each electron.
MIXTURES = (0.1, 0.25, 0.5)


# --- math -----------------------------------------------------------------------------
def mass_shell(extent: float, count: int):
    """The Hamiltonian's eigenvalues over a grid of momenta in the xy plane."""
    along = np.linspace(-extent, extent, count)
    px, py = np.meshgrid(along, along)
    momenta = mv(core.Spatial, np.stack([px, py, np.zeros_like(px)], axis=-1))  # [count, count] Spatial
    values, states = core.hamiltonian(momenta, MASS).eigh()                     # [count, count, 8] each

    # --- checks
    # Each momentum has the energies minus and plus `E`, each fourfold; the positive-energy states
    # have `beta == 0` and the negative-energy states `beta == np.pi`, where `core.invariants(states)`
    # has a negative scalar part.
    E = core.energy(momenta, MASS).to_array()[..., None]
    np.testing.assert_allclose(values.to_array(), np.concatenate([-E.repeat(4, -1), E.repeat(4, -1)], -1), atol=1e-12)
    scalar = core.invariants(states).select[0].to_array()
    np.testing.assert_allclose(np.sign(scalar), np.concatenate([-np.ones(4), np.ones(4)]) * np.ones_like(scalar))
    return momenta, values


def trembling(momentum: core.Vector, seconds: float, count: int):
    """Electrons of the given momentum with a share of negative energy mixed in, followed in time:
    their spinors, and the paths their currents trace."""
    H = core.hamiltonian(momentum, MASS)
    E = core.energy(momentum, MASS)
    # A positive-energy state with spin along z, and a negative-energy state that turns the other way.
    electron = 0.5 * (mv.scalar([1.0]) + H(mv.scalar([1.0])) / E)
    positron = 0.5 * (mv.tx - H(mv.tx) / E)
    electron, positron = electron / core.invariants(electron).select[0].square_root(), positron / (-core.invariants(positron).select[0]).square_root()
    times = np.linspace(0.0, seconds, count)
    spinors, paths = [], []
    for share in MIXTURES:
        psi = core.evolve(momentum, MASS, electron * np.sqrt(1 - share) + positron * np.sqrt(share), times)   # [times] Spinor
        spinors.append(psi)
        paths.append(core.path(core.velocity(core.current(psi)), times[1] - times[0]))                   # [times] Bivector

    # --- checks
    # The density the current carries is constant in time, and the Hamiltonian commutes with
    # right multiplication by the spin plane.
    for psi in spinors:
        density = (core.current(psi) | core.TIME).to_array()
        np.testing.assert_allclose(density, density[0], rtol=1e-10)
    probe = mv(core.Even.output_subspace, np.random.default_rng(0).normal(size=8))
    unit = core.Even * core.SPIN_PLANE
    np.testing.assert_allclose(H(unit(probe)).kernel, unit(H(probe)).kernel, atol=1e-12)
    return times, spinors, paths


if __name__ == "__main__":
    from examples.animation import save_animation, save_figure
    from examples.relativity.dirac import render

    save_figure(render.draw_mass_shell(*mass_shell(2.0, 41), MASS), "dirac_mass_shell")
    times, spinors, paths = trembling(mv(core.Spatial, np.array([0.0, 0.0, 0.3])), 12.0, 600)
    save_figure(render.draw_paths(times, spinors, paths, MIXTURES), "dirac_trembling")
    save_animation(render.animate_paths(times, spinors, paths, MIXTURES, 6), "dirac_trembling", 40)
