"""Corner hats and edge forms on spacetime, and the Lorentzian dual; the wake and the mirror's wave run on coarse
strips and draw."""

import matplotlib.pyplot as plt
import numpy as np

from examples.relativity.moving_mirror import core, render, scenarios


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
    tails, heads = core.triangle.corners[2, 1]
    classic_forms = (hats[tails] * gradients[heads, None] - hats[heads] * gradients[tails, None]) * core.triangle.orientation[1].T
    np.testing.assert_allclose((edge_forms - classic_forms).kernel, 0.0, atol=1e-14)

    sides = (core.triangle.boundary[2, 1] * (core.triangle.d[0] * corners.field())).batch()[:, 0]
    for fraction in (0.2, 0.5, 0.85):
        edge_events = corners[tails] * (1 - fraction) + corners[heads] * fraction
        readings = core.edge_whitney_forms(corners, edge_events) & sides[None, :]
        np.testing.assert_allclose(readings.kernel, np.eye(3)[..., None], atol=1e-14)

    # A null gradient has zero pairing with its Lorentzian dual; observer energy stays positive.
    right_null, left_null = core.mv.x - core.mv.t, core.mv.x + core.mv.t
    np.testing.assert_allclose((right_null & core.Plane.dual()(right_null)).kernel, 0.0, atol=1e-14)
    np.testing.assert_allclose((left_null & core.Plane.dual()(left_null)).kernel, 0.0, atol=1e-14)
    plane = 3.0 * core.mv.x + 4.0 * core.mv.t
    np.testing.assert_allclose((plane & core.Plane.dual()(plane)).kernel, -7.0, atol=1e-14)
    np.testing.assert_allclose((-(plane & (core.mv.t >> core.Plane.dual()(plane)))).kernel, 25.0, atol=1e-14)


def test_the_wake_and_the_mirror_wave_run_and_draw():
    n_space, n_time = 5, 5
    corners, points = scenarios.single_triangle(core.space, core.time, 2)
    _, gradients, _ = core.barycentric_hats(corners, points)
    edge_forms = core.edge_whitney_forms(corners, points)
    wake_topology, wake_vertices, _ = scenarios.spacetime_strip(
        core.space, core.time, n_space, n_time, np.ones(n_time), scenarios.wake_duration)
    source_event = scenarios.source_x * core.space + scenarios.source_t * core.time
    wake = core.causal_wake(core.Operators(wake_topology, wake_vertices), n_space, n_time, source_event,
                            scenarios.source_width)
    plt.close(render.draw_light_cone_whitney(corners, points, gradients, edge_forms, wake_topology, wake_vertices,
                                             wake, source_event))

    # The cavity animation scales its colours by the wave at row 20.
    n_time = 24
    times = np.linspace(0.0, scenarios.mirror_duration, n_time)
    wall_positions = scenarios.mirror_trajectory(times, scenarios.initial_length, scenarios.stroke,
                                                 scenarios.inflection_time, scenarios.transition_time)
    topology, vertices, dt = scenarios.spacetime_strip(
        core.space, core.time, n_space, n_time, wall_positions, scenarios.mirror_duration)
    _, _, wave_rate, slice_energies = core.moving_mirror_wave(
        core.Operators(topology, vertices), n_space, n_time, dt,
        scenarios.pulse_centre, scenarios.pulse_width, scenarios.wavelength)
    plt.close(render.draw_moving_mirror_doppler(topology, vertices, n_space, n_time, wave_rate, slice_energies, 1, 3))
    assert len(list(render.animate_moving_mirror(topology, vertices, n_space, n_time, wave_rate, np.array([2])))) == 1
