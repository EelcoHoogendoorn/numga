"""Spacetime Whitney forms, the Lorentzian dual, causal light-cone wakes, and relativistic moving-mirror Doppler shift."""

import numpy as np

from examples.relativity.spacetime_whitney import core, scenarios


def test_spacetime_barycentric_hats_edge_forms_and_lorentzian_dual():
    corners, points = scenarios.single_triangle(core.space, core.time, 7)
    hats, gradients, patch = core.barycentric_hats(corners, points)
    corner_hats, _, _ = core.barycentric_hats(corners, corners)

    assert hats.gatype == core.Scalar
    assert gradients.gatype == core.Plane
    assert patch.gatype == core.Patch
    np.testing.assert_allclose(corner_hats.kernel, np.eye(3)[..., None], atol=1e-14)
    np.testing.assert_allclose(hats.sum(axis=0).kernel, 1.0, atol=1e-14)
    np.testing.assert_allclose(gradients.sum(axis=0).kernel, 0.0, atol=1e-14)

    # Barycentric interpolation reproduces any affine spacetime potential and its 1-form gradient exactly.
    true_gradient = 1.3 * core.mv.x - 0.75 * core.mv.t
    corner_values = 0.4 + (true_gradient & corners)
    sample_values = 0.4 + (true_gradient & points)
    interpolated = (corner_values[:, None] * hats).sum(axis=0)
    reconstructed_gradient = (corner_values * gradients).sum(axis=0)
    np.testing.assert_allclose(interpolated.kernel, sample_values.kernel, atol=1e-14)
    np.testing.assert_allclose(reconstructed_gradient.kernel, true_gradient.kernel, atol=1e-14)

    # Edge 1-forms match Whitney's alternating formula and satisfy Kronecker duality along the worldline edges.
    edge_forms = core.edge_whitney_forms(corners, points)
    first, second = [1, 2, 0], [2, 0, 1]
    classic_forms = hats[first] * gradients[second, None] - hats[second] * gradients[first, None]
    np.testing.assert_allclose((edge_forms - classic_forms).kernel, 0.0, atol=1e-14)

    sides = corners[[2, 0, 1]] - corners[[1, 2, 0]]
    for fraction in (0.2, 0.5, 0.85):
        edge_events = corners[first] * (1 - fraction) + corners[second] * fraction
        readings = core.edge_whitney_forms(corners, edge_events) & sides[None, :]
        np.testing.assert_allclose(readings.kernel, np.eye(3)[..., None], atol=1e-14)

    # A null gradient has zero pairing with its Lorentzian dual; observer energy stays positive.
    right_null, left_null = core.mv.x - core.mv.t, core.mv.x + core.mv.t
    np.testing.assert_allclose((right_null & core.Plane.dual()(right_null)).kernel, 0.0, atol=1e-14)
    np.testing.assert_allclose((left_null & core.Plane.dual()(left_null)).kernel, 0.0, atol=1e-14)
    plane = 3.0 * core.mv.x + 4.0 * core.mv.t
    np.testing.assert_allclose((plane & core.Plane.dual()(plane)).kernel, -7.0, atol=1e-14)
    np.testing.assert_allclose((-(plane & (core.mv.t >> core.Plane.dual()(plane)))).kernel, 25.0, atol=1e-14)


def test_lorentzian_wave_operator_and_causal_light_cone_wake():
    n_space, n_time = 45, 75
    duration = 0.72
    mesh, vertices, _ = scenarios.spacetime_strip(
        core.space, core.time, n_space, n_time, np.ones(n_time), duration,
    )
    operators = core.Operators(mesh, vertices)

    assert operators.gradient.gatype == core.ga.gatype((core.Plane, core.Scalar))
    assert operators.wave_operator.gatype == core.ga.gatype((core.Amount, core.Scalar))

    # The wave operator is symmetric.
    np.testing.assert_allclose(
        (operators.wave_operator - operators.wave_operator.adjugate()).cells.kernel,
        0.0, atol=1e-12,
    )

    # Causal light-cone wake from a localized event vanishes outside the forward light cone.
    source_event = 0.5 * core.space + 0.14 * core.time
    wake = core.causal_wake(operators, n_space, n_time, source_event, 0.042)
    displacements = vertices.batch() - source_event
    distances = (core.mv.x & displacements).norm()
    delays = core.mv.t & displacements
    outside_cone = (distances > delays + 0.16) | (delays < -0.12)
    wake_batch = wake.batch()
    assert np.max(np.abs(wake_batch[outside_cone].kernel)) < 0.025 * np.max(np.abs(wake_batch.kernel))


def test_relativistic_moving_mirror_doppler_compression_and_energy_gain():
    n_space, n_time = 65, 300
    duration = 1.25
    times = np.linspace(0.0, duration, n_time)
    wall_positions = scenarios.mirror_trajectory(
        times, scenarios.initial_length, scenarios.stroke,
        scenarios.inflection_time, scenarios.transition_time,
    )
    mesh, vertices, dt = scenarios.spacetime_strip(
        core.space, core.time, n_space, n_time, wall_positions, duration,
    )
    operators = core.Operators(mesh, vertices)
    potential, field, wave_rate, slice_energies = core.moving_mirror_wave(
        operators, n_space, n_time, dt,
        scenarios.pulse_centre, scenarios.pulse_width, scenarios.wavelength,
    )

    assert potential.batch().gatype == core.Scalar
    assert field.batch().gatype == core.Plane
    assert wave_rate.batch().gatype == core.Scalar
    assert slice_energies.gatype == core.Scalar

    # Dirichlet boundary condition phi = 0 holds identically along both mirror worldlines.
    potential_grid = potential.batch().reshape(n_time, n_space)
    np.testing.assert_allclose(potential_grid[:, 0].kernel, 0.0, atol=1e-14)
    np.testing.assert_allclose(potential_grid[:, -1].kernel, 0.0, atol=1e-14)

    # Euler-Lagrange stationarity holds at every interior spacetime vertex.
    grid = np.arange(n_time * n_space).reshape(n_time, n_space)
    interior = ~core.SparseExtensor.selection(core.mv.context, grid[1:-1, 1:-1].ravel(), mesh.n_vertices) * core.Scalar
    np.testing.assert_allclose(interior.adjugate()(operators.wave_operator(potential)).kernel, 0.0, atol=1e-10)

    # Wave energy is conserved before and after reflection and boosted by the Einstein Doppler factor across reflection.
    np.testing.assert_allclose(
        (slice_energies[45] / slice_energies[15]).kernel, 1.0, rtol=2e-2,
    )
    np.testing.assert_allclose(
        (slice_energies[260] / slice_energies[230]).kernel, 1.0, rtol=2e-2,
    )
    wave_rate_grid = wave_rate.batch().reshape(n_time, n_space)
    assert np.max(np.abs(wave_rate_grid[240].kernel)) > 1.70 * np.max(np.abs(wave_rate_grid[45].kernel))
    assert slice_energies[240] > 2.10 * slice_energies[45]
