"""Whitney commutation, the flux-preserving subspace, and the Beltrami fields of a solid torus."""

import numpy as np
from numga import stack

from examples.electromagnetism.beltrami_fields import core, numerical, scenarios

# At rest in one instant of spacetime, the gradients are planes of space.
SpatialPlane = core.Plane.derive.within("x y z")


def test_whitney_commutation_flux_preserving_subspace_and_chiral_beltrami_spectrum():
    mesh, vertices, free = scenarios.torus_cavity()
    operators = core.Operators(mesh, vertices)

    assert operators.whitney1.gatype == core.ga.gatype((SpatialPlane, core.Scalar))
    assert operators.curl.gatype == core.ga.gatype((core.Field, core.Scalar))
    assert operators.stiffness.gatype == core.ga.gatype((core.Amount, core.Scalar))
    assert operators.helicity.gatype == core.ga.gatype((core.Amount, core.Scalar))
    np.testing.assert_allclose((operators.stiffness - operators.stiffness.adjugate()).cells.kernel, 0.0, atol=1e-12)
    np.testing.assert_allclose((operators.helicity - operators.helicity.adjugate()).cells.kernel, 0.0, atol=1e-14)

    # Whitney reconstruction commutes with the exterior derivative.
    d0, d1 = mesh.d[0] * core.Scalar, mesh.d[1] * core.Scalar
    potential = (core.mv.x & vertices) * 0.7 - (core.mv.y & vertices) * 1.1 + (core.mv.z & vertices) * 0.5
    np.testing.assert_allclose((operators.whitney1(d0(potential)) - operators.gradient(potential)).kernel, 0.0, atol=1e-12)
    free_test = free(core.as_scalar(np.sin(np.arange(free.shape[1]))).field())
    np.testing.assert_allclose((operators.curl(free_test) - operators.whitney2(d1(free_test))).kernel, 0.0, atol=1e-11)
    # Normal magnetic flux d1(free_test) vanishes identically on every boundary triangle of the torus.
    np.testing.assert_allclose(d1(free_test).batch()[mesh.exterior(2)].kernel, 0.0, atol=1e-14)

    # An interior vertex gradient lies in the kernel of both stiffness and free-subspace helicity.
    interior_phi = core.as_scalar(np.arange(mesh.n_vertices) < scenarios.n_major).field()
    gauge_edge = d0(interior_phi)
    np.testing.assert_allclose(operators.stiffness(gauge_edge).kernel, 0.0, atol=1e-12)
    np.testing.assert_allclose(free.adjugate()(operators.helicity(gauge_edge)).kernel, 0.0, atol=1e-13)

    # Chiral symmetry and Beltrami eigenmode residuals.
    inverse_eigenvalues, modes, n_null = numerical.coexact_helicity_eigh(
        free.adjugate()(operators.stiffness(free)),
        free.adjugate()(operators.helicity(free)),
    )
    modes = free(modes)
    assert n_null == scenarios.n_major
    np.testing.assert_allclose((inverse_eigenvalues + inverse_eigenvalues[::-1]).kernel, 0.0, atol=1e-13)

    check_indices = np.array([0, 1, -2, -1])
    check_modes = modes[check_indices]
    check_eigenvalues = 1 / inverse_eigenvalues[check_indices]
    magnetic = operators.curl(check_modes)
    energies = (operators.stiffness(check_modes) & check_modes).sites.sum() / 2
    helicities = (operators.helicity(check_modes) & check_modes).sites.sum()
    residuals = free.adjugate()(operators.stiffness(check_modes) - check_eigenvalues * operators.helicity(check_modes))
    assert magnetic.batch().gatype == core.Field
    np.testing.assert_allclose(energies.kernel[..., 0], 0.5, atol=1e-12)
    np.testing.assert_allclose((helicities - inverse_eigenvalues[check_indices]).kernel, 0.0, atol=1e-12)
    np.testing.assert_allclose(residuals.kernel, 0.0, atol=1e-11)


def test_selective_decay_helicity_conservation_and_woltjer_taylor_relaxation():
    mesh, vertices, free = scenarios.torus_cavity()
    operators = core.Operators(mesh, vertices)
    inverse_eigenvalues, modes, _ = numerical.coexact_helicity_eigh(
        free.adjugate()(operators.stiffness(free)),
        free.adjugate()(operators.helicity(free)),
    )
    modes = free(modes)

    selected_modes = modes[scenarios.selected_indices]
    selected_eigenvalues = 1 / inverse_eigenvalues[scenarios.selected_indices]
    initial_amplitudes = core.as_scalar(scenarios.initial_weights)
    times = np.linspace(0.0, scenarios.relaxation_time, scenarios.n_frames)
    states = list(core.selective_decay(initial_amplitudes, selected_eigenvalues, times))
    amplitude_frames = stack(states)
    energy_frames = amplitude_frames.squared().sum(axis=1) / 2
    helicity_frames = (amplitude_frames.squared() / selected_eigenvalues).sum(axis=1)
    fields = core.reconstruct(iter(states), selected_modes, operators.curl)
    for amplitudes, field in zip(states, fields, strict=True):
        direct_field = operators.curl((amplitudes * selected_modes).sum(axis=0))
        np.testing.assert_allclose((field - direct_field).kernel, 0.0, atol=1e-12)
    np.testing.assert_allclose((helicity_frames - helicity_frames[0]).kernel, 0.0, atol=1e-12)
    assert np.all(np.diff(energy_frames.kernel[:, 0]) < 0)

    taylor_bound = (helicity_frames[0] * selected_eigenvalues[0] / 2).kernel[0]
    assert energy_frames[0].kernel[0] > 1.8 * taylor_bound
    assert energy_frames[-1].kernel[0] > taylor_bound
    assert abs(energy_frames[-1].kernel[0] - taylor_bound) / taylor_bound < 0.01

    # Verify that the final reconstructed field matches the direct operator energy and helicity on the mesh.
    final_cochain = (amplitude_frames[-1] * selected_modes).sum(axis=0)
    direct_energy = (operators.stiffness(final_cochain) & final_cochain).sites.sum() / 2
    direct_helicity = (operators.helicity(final_cochain) & final_cochain).sites.sum()
    np.testing.assert_allclose(direct_energy.kernel, energy_frames[-1].kernel, atol=1e-12)
    np.testing.assert_allclose(direct_helicity.kernel, helicity_frames[0].kernel, atol=1e-12)
