"""Natural-unit Zitterbewegung scenarios follow Dirac electrons of fixed spatial momentum across
varying negative-energy admixtures.

With the reduced Planck constant, the speed of light, and the electron rest mass all set to one,
lengths are measured in reduced Compton wavelengths (386 femtometres) and times in the light-crossing
time of one reduced Compton wavelength.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import stack
from examples.relativity.dirac import core

mv = core.mv
mass = 1.0
# Negative-energy probability shares for the blue, purple, and red trajectories.
mixtures = np.array([0.1, 0.25, 0.5])
# Drift momentum along +z, total integration span, frame count, substeps per frame, and frame delay.
drift_coordinates = [0.0, 0.0, 0.3]
duration_seconds = 12.0
n_frames, n_substeps = 100, 6
n_times = n_frames * n_substeps
dt = duration_seconds / (n_times - 1)
frame_duration_ms = 40


# --- math -----------------------------------------------------------------------------
def zitterbewegung_states() -> Iterator[tuple[core.Antivector, core.Spinor]]:
    """Yields `(trail, spinor)` states (`[3, ...] Antivector-`, `[3] Spinor+`) for the three
    negative-energy shares drifting along `drift_coordinates`."""
    momentum = mv(core.Spatial, drift_coordinates)                              # [] Vector+
    initial = core.mixed_spinors(momentum, mass, mixtures)                      # [3] Spinor+
    return core.trembling_states(momentum, mass, initial, dt, n_frames, n_substeps)


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.relativity.dirac import render
    from examples.util.animation import save_animation, save_figure

    recorded: list[tuple[core.Antivector, core.Spinor]] = []

    def streaming() -> Iterator[tuple[core.Antivector, core.Spinor]]:
        for state in zitterbewegung_states():
            recorded.append(state)
            yield state

    save_animation(render.paths_animation(streaming()), "dirac_trembling", frame_duration_ms)
    save_figure(render.paths_figure(recorded), "dirac_trembling")
    final_paths = recorded[-1][0]                                               # [3, n_times] Antivector-

    # checks
    # Charge density across the observer's slice stays constant across frames, the 50% admixture has
    # zero net z-drift, and the plane-wave Hamiltonian commutes with right multiplication by spin_plane.
    frame_spinors = stack([psi for _, psi in recorded], axis=-1)                # [3, n_frames] Spinor+
    densities = mv.t & (frame_spinors >> core.time)                             # [3, n_frames] Scalar+
    assert ((densities - densities[:, :1]).abs() < 1e-10).all()
    assert ((mv.z & final_paths[-1]).abs() < 1e-2).all()

    momentum = mv(core.Spatial, drift_coordinates)                              # [] Vector+
    hamiltonian = core.plane_wave_hamiltonian(momentum, mass)                   # [] Spinor+ <- Spinor+
    probe = mv(core.Spinor, np.random.default_rng(0).normal(size=8))            # [] Spinor+
    commutator_residual = hamiltonian(core.phase_unit(probe)) - core.phase_unit(hamiltonian(probe))
    residual_norm = commutator_residual.scalar_product(mv.t * commutator_residual.reverse() * mv.t.inverse())
    assert (residual_norm.abs() < 1e-12).all()


if __name__ == "__main__":
    main()
