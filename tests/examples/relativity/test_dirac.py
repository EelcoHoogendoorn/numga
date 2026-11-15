"""Coverage tests verify Dirac spinor frame action, bivector duality, plane-wave energies, and
Zitterbewegung rendering."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.relativity.dirac import core, render, scenarios


def test_spinor_vector_frame_and_bivector_duality():
    """Sandwiching vectors scales the Lorentz frame by the density, while `bivector_duality` isolates
    `(pseudoscalar * beta).exp() / density` on bivectors."""
    mv = core.mv
    density, beta = 2.0, 0.7
    rotor = ((mv.tx * 0.4 + mv.xy * 0.9 + mv.yz * -0.3) * 0.5).exp()            # [] Spinor+
    psi = core.dirac_spinor(density, beta, rotor)                               # [] Spinor+
    vector = mv.vector([0.3, -1.2, 0.5, 2.0])                                   # [] Vector+
    frame_diff = (psi >> vector) - density * (rotor >> vector)                  # [] Vector+
    vector_norm = frame_diff.scalar_product(mv.t * frame_diff.reverse() * mv.t)
    assert (vector_norm.abs() < 1e-12).all()

    plane = mv.bivector([0.2, -0.4, 1.1, 0.3, -0.7, 0.5])                       # [] Bivector+
    expected = ((core.pseudoscalar * beta).exp() / density) * plane             # [] Bivector+
    duality_diff = core.bivector_duality(psi)(plane) - expected                 # [] Bivector+
    bivector_norm = duality_diff.scalar_product(mv.t * duality_diff.reverse() * mv.t.inverse())
    assert (bivector_norm.abs() < 1e-12).all()


def test_plane_wave_energies_and_beta_signs():
    """Plane-wave Hamiltonians have fourfold `-E` and `+E` eigenvalues whose scalar reverse norms
    carry negative and positive signs."""
    grid = np.linspace(-1.0, 1.0, 3)
    px, py = np.meshgrid(grid, grid)
    momenta = core.mv(core.Spatial, np.stack([px, py, np.zeros_like(px)], axis=-1))
    values, states = core.plane_wave_hamiltonian(momenta, scenarios.mass).eigh()
    energy = core.plane_wave_energy(momenta, scenarios.mass)                    # [3, 3] Scalar+
    assert ((values[..., :4] + energy[..., None]).abs() < 1e-12).all()
    assert ((values[..., 4:] - energy[..., None]).abs() < 1e-12).all()
    signs = states.scalar_norm_squared()                                        # [3, 3, 8] Scalar+
    assert (signs[..., :4] < 0).all()
    assert (signs[..., 4:] > 0).all()


def test_trembling_trajectory_figure_and_animation():
    """Single-frame Zitterbewegung states render cleanly through both `paths_figure` and
    `paths_animation`."""
    momentum = core.mv(core.Spatial, scenarios.drift_coordinates)               # [] Vector+
    initial = core.mixed_spinors(momentum, scenarios.mass, scenarios.mixtures)  # [3] Spinor+
    states = list(core.trembling_states(momentum, scenarios.mass, initial, 0.1, 1, 2))
    plt.close(render.paths_figure(states))
    frame = next(render.paths_animation(states))
    assert frame.ndim == 3 and frame.shape[-1] == 3
