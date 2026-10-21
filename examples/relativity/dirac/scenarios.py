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
MIXTURES = np.array([0.1, 0.25, 0.5])
# The electrons' momentum along z, how long they are followed and in how many samples; the samples between
# animation frames, and a frame's duration in milliseconds.
MOMENTUM = [0.0, 0.0, 0.3]
SECONDS, N_TIMES = 12.0, 600
FRAME_STRIDE, FRAME_DURATION = 6, 40


# --- math -----------------------------------------------------------------------------
def trembling(momentum: core.Vector, seconds: float, n_times: int):
    """Electrons of the given momentum with a share of negative energy mixed in, followed in time:
    the times, their spinors and the paths their currents trace, `[mixtures, times]`, and the
    plane-wave Hamiltonian."""
    hamiltonian = core.hamiltonian(momentum, MASS)                              # [] Even <- Even
    energy = core.energy(momentum, MASS)                                        # [] Scalar
    one = mv.scalar([1.0])
    # A positive-energy state with spin along z, and a negative-energy state that turns the other way,
    # each scaled to unit density.
    electron = 0.5 * (one + hamiltonian(one) / energy)                          # [] Spinor
    positron = 0.5 * (mv.tx - hamiltonian(mv.tx) / energy)                      # [] Spinor
    electron, positron = electron / electron.scalar_norm_squared().square_root(), positron / (-positron.scalar_norm_squared()).square_root()
    times = np.linspace(0.0, seconds, n_times)
    mixed = electron * np.sqrt(1 - MIXTURES) + positron * np.sqrt(MIXTURES)      # [mixtures] Spinor
    spinors = core.evolve(momentum, MASS, mixed[:, None], times)                 # [mixtures, times] Spinor
    paths = core.path(core.velocity(core.current(spinors)), times[1] - times[0])  # [mixtures, times] Bivector
    return times, spinors, paths, hamiltonian


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.animation import save_animation, save_figure
    from examples.relativity.dirac import render

    times, spinors, paths, hamiltonian = trembling(mv(core.Spatial, MOMENTUM), SECONDS, N_TIMES)
    save_figure(render.draw_paths(times, spinors, paths, MIXTURES), "dirac_trembling")
    save_animation(render.animate_paths(times, spinors, paths, MIXTURES, FRAME_STRIDE), "dirac_trembling", FRAME_DURATION)

    # checks
    # The density the current carries is constant in time, and the Hamiltonian commutes with right
    # multiplication by the spin plane.
    density = (core.current(spinors) | core.TIME).to_array()                    # [mixtures, times]
    np.testing.assert_allclose(density, density[:, :1] * np.ones_like(density), rtol=1e-10)
    probe = mv(core.Even, np.random.default_rng(0).normal(size=8))
    unit = core.Even * core.SPIN_PLANE
    np.testing.assert_allclose(hamiltonian(unit(probe)).kernel, unit(hamiltonian(probe)).kernel, atol=1e-12)


if __name__ == "__main__":
    main()
