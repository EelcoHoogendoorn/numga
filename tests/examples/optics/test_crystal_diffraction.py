"""The inverse adjoint preserves phases and the finite crystal's scattering pattern."""

import numpy as np

from examples.optics.crystal_diffraction import core


def test_deforming_atoms_and_phase_measurements_preserves_scattering():
    shells, orders, spacing = 2, 2, 1.3
    stretches = np.array([0.8, 1.7, 1.2])
    shears = np.array([-0.3, 0.6, 0.4])
    angles = np.array([0.2, -0.4, 0.7])
    reference = core.square(shells, orders, spacing)
    deformation = core.deformation(stretches, shears, angles)
    crystals = reference.carried(deformation)
    grid = core.transfer_grid(8.0, 17)
    moved_grid = deformation.adjoint()[:, None, None].solve(grid)
    residual = crystals.intensity(moved_grid) - reference.intensity(grid)
    phases = crystals.reciprocal[:, :, None] | crystals.positions[:, None, :]
    reference_phases = reference.reciprocal[:, None] | reference.positions[None, :]
    peaks = crystals.intensity(crystals.reciprocal[:, None, :])

    np.testing.assert_allclose(residual.kernel, 0, atol=1e-11)
    np.testing.assert_allclose((phases - reference_phases).kernel, 0, atol=1e-10)
    np.testing.assert_allclose((crystals.phase_planes - crystals.reciprocal.dual()).kernel, 0, atol=1e-11)
    np.testing.assert_allclose(peaks.kernel, 1, atol=1e-12)
