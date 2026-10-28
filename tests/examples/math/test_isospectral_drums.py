"""Invertible transplantation, matching seams and the complete discrete wave operator."""

import numpy as np
import pytest

from numga.sparse import SparseExtensor
from examples.math.isospectral_drums import core, numerics, scenarios


round_off = 1e-10


@pytest.mark.parametrize("n_subdivisions", [4, 5, 6, 7, 8])
def test_transplantation_is_invertible_and_intertwines_every_field(n_subdivisions):
    _, _, source_neighbors, source_restriction, _, _ = numerics.drum(
        scenarios.source_indices, scenarios.n_lattice, n_subdivisions)
    _, _, target_neighbors, target_restriction, target_assembly, _ = numerics.drum(
        scenarios.target_indices, scenarios.n_lattice, n_subdivisions)
    n_samples = (n_subdivisions + 1) * (n_subdivisions + 2) // 2
    mixing = core.tile_mixing(scenarios.contributors, scenarios.signs, n_samples)
    transplant = target_assembly(mixing(source_restriction))
    source_laplacian = (4 * core.Scalar - source_neighbors) * n_subdivisions ** 2
    target_laplacian = (4 * core.Scalar - target_neighbors) * n_subdivisions ** 2

    # Every nodal impulse is recovered, so no field can lie in the transplant's kernel.
    n_interior = source_laplacian.shape[0]
    impulses = core.mv.scalar(np.eye(n_interior)[..., None]).field()
    restored = transplant.solve(transplant(impulses))
    commutator = target_laplacian(transplant) - transplant(source_laplacian)
    seams = target_restriction(transplant) - mixing(source_restriction)

    # checks
    np.testing.assert_allclose(restored.kernel, impulses.kernel, atol=round_off, rtol=0)
    np.testing.assert_allclose(commutator.cells.kernel, 0, atol=round_off, rtol=0)
    np.testing.assert_allclose(seams.cells.kernel, 0, atol=round_off, rtol=0)


def test_independent_eigensolves_and_transplanted_modes_have_the_same_frequencies():
    n_subdivisions, n_modes = 12, 10
    _, _, source_neighbors, source_restriction, _, _ = numerics.drum(
        scenarios.source_indices, scenarios.n_lattice, n_subdivisions)
    _, _, target_neighbors, _, target_assembly, _ = numerics.drum(
        scenarios.target_indices, scenarios.n_lattice, n_subdivisions)
    n_samples = (n_subdivisions + 1) * (n_subdivisions + 2) // 2
    transplant = target_assembly(core.tile_mixing(
        scenarios.contributors, scenarios.signs, n_samples)(source_restriction))
    source_laplacian = (4 * core.Scalar - source_neighbors) * n_subdivisions ** 2
    target_laplacian = (4 * core.Scalar - target_neighbors) * n_subdivisions ** 2
    n_interior = source_laplacian.shape[0]
    identity = SparseExtensor.selection(core.context, np.arange(n_interior), n_interior) * core.Scalar
    source_values, source_modes = source_laplacian.eigh(identity, n_modes)
    target_values, _ = target_laplacian.eigh(identity, n_modes)
    target_modes = transplant(source_modes)
    residual = target_laplacian(target_modes) - target_modes * source_values

    # checks
    np.testing.assert_allclose(source_values.kernel, target_values.kernel, atol=round_off, rtol=0)
    np.testing.assert_allclose(residual.kernel, 0, atol=round_off, rtol=0)
