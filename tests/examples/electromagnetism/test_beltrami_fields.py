"""Tetrahedral Whitney 0-, 1-, and 2-forms, topological magnetic helicity, and Beltrami fields."""

import numpy as np
from numga import stack

from examples.electromagnetism.beltrami_fields import core, numerical, plumbing, scenarios

# At rest in one instant of spacetime, the gradients and their wedges are planes and volumes of space,
# and the patches and volumes of the geometry span directions of space.
SpatialPlane = core.Plane.derive.within("x y z")
SpatialTrivector = core.Trivector.derive.within("x y z")
SpatialPatch = core.Patch.derive.containing("t")
SpatialVolume = core.Volume.derive.containing("t")


def test_tetrahedral_whitney_0_1_2_forms_and_topological_opposite_edge_helicity():
    mesh, vertices, points = scenarios.single_tetrahedron(6)
    corners = mesh.incidence[3, 0] * vertices
    hats, gradients, face_patches, volume = core.tetrahedral_hats(mesh, vertices, points)
    corner_hats, _, _, _ = core.tetrahedral_hats(mesh, vertices, corners)

    assert hats.batch().gatype == core.Scalar
    assert gradients.batch().gatype == SpatialPlane
    assert face_patches.batch().gatype == SpatialPatch
    assert volume.batch().gatype == SpatialVolume
    np.testing.assert_allclose(corner_hats.kernel, np.eye(4)[..., None, None], atol=1e-13)
    np.testing.assert_allclose(hats.sum(axis=0).kernel, 1.0, atol=1e-13)
    np.testing.assert_allclose(gradients.sum(axis=0).kernel, 0.0, atol=1e-13)
    np.testing.assert_allclose(face_patches.sum(axis=0).kernel, 0.0, atol=1e-14)

    # Barycentric interpolation reproduces any affine 3D scalar potential and its gradient exactly.
    true_gradient = 1.3 * core.mv.x - 0.75 * core.mv.y + 0.55 * core.mv.z
    corner_values = 0.4 + (true_gradient & corners)
    sample_values = 0.4 + (true_gradient & points)
    interpolated = (corner_values[:, None] * hats).sum(axis=0)
    reconstructed_gradient = (corner_values * gradients).sum(axis=0)
    np.testing.assert_allclose(interpolated.kernel, sample_values.kernel, atol=1e-13)
    np.testing.assert_allclose((reconstructed_gradient - true_gradient).kernel, 0.0, atol=1e-13)

    # Edge Whitney 1-forms match Whitney's alternating formula, have constant curl, and satisfy Kronecker duality on all 6 edges.
    edge_forms, edge_curls = core.edge_whitney_forms(mesh, vertices, points)
    tails, heads = mesh.corners(3, 1).T
    classic_1forms = hats[tails] * gradients[heads, None] - hats[heads] * gradients[tails, None]
    classic_curls = 2 * (gradients[tails] ^ gradients[heads])
    np.testing.assert_allclose((edge_forms - classic_1forms).kernel, 0.0, atol=1e-13)
    np.testing.assert_allclose((edge_curls - classic_curls).kernel, 0.0, atol=1e-13)

    edge_vectors = mesh.oriented[3, 1] * (mesh.d[0] * vertices)
    for fraction in (0.2, 0.5, 0.85):
        edge_pts = corners[tails] * (1 - fraction) + corners[heads] * fraction
        readings, _ = core.edge_whitney_forms(mesh, vertices, edge_pts)
        np.testing.assert_allclose((readings & edge_vectors[None, :]).kernel, np.eye(6)[..., None, None], atol=1e-13)

    # Face Whitney 2-forms match Whitney's 2-form formula and integrate to Kronecker delta across the 4 faces.
    face_forms = core.face_whitney_forms(mesh, vertices, points)
    first, second, third = plumbing.complements(mesh, 0).T
    li, lj, ll = hats[first], hats[second], hats[third]
    gi, gj, gl = gradients[first, None], gradients[second, None], gradients[third, None]
    classic_2forms = 2 * (li * (gj ^ gl) + lj * (gl ^ gi) + ll * (gi ^ gj))
    np.testing.assert_allclose((face_forms - classic_2forms).kernel, 0.0, atol=1e-13)

    face_centres = mesh.incidence[3, 2] * (mesh.incidence[2, 0].mean(axis=0) * vertices)
    face_readings = core.face_whitney_forms(mesh, vertices, face_centres) & face_patches[None, :]
    np.testing.assert_allclose(face_readings.kernel, np.eye(4)[..., None, None], atol=1e-14)

    # The 6x6 tetrahedral helicity matrix is purely topological, invariant under arbitrary affine deformation,
    # and pairs only opposite edges with linking weight +-1/6.
    tet_helicity, tet_densities = core.tetrahedral_helicity(mesh, vertices)
    deformed_helicity, _ = core.tetrahedral_helicity(mesh, scenarios.deformed_tetrahedron())
    assert tet_helicity.batch().gatype == core.Scalar
    assert tet_densities.batch().gatype == SpatialTrivector
    np.testing.assert_allclose((tet_helicity - deformed_helicity).kernel, 0.0, atol=1e-14)
    expected_linking = np.zeros((6, 6))
    expected_linking[np.arange(6), 5 - np.arange(6)] = np.array([1, -1, 1, 1, -1, 1]) / 6
    np.testing.assert_allclose(tet_helicity.kernel[..., 0, 0], expected_linking, atol=1e-14)


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
