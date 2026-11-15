"""Tests for spacetime simplex gradients, edge 1-forms, causal light-cone wakes, and moving-mirror waves."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.relativity.moving_mirror import core, render, scenarios


def test_spacetime_simplex_gradients_edge_forms_and_lorentzian_dual():
    """Simplex gradients, edge 1-forms, and Lorentzian dual pairings satisfy partition, Kronecker, and null identities."""
    corners, points = scenarios.single_triangle(core.space, core.time, 5)
    weights, gradients, patch = core.corner_interpolation(corners, points)
    corner_events = corners.batch()
    corner_weights, _, _ = core.corner_interpolation(corners, corner_events)

    assert weights.gatype == core.Scalar
    assert gradients.gatype == core.Plane
    assert patch.gatype == core.Patch
    assert ((corner_weights - np.eye(3)).abs() < 1e-10).all()
    assert ((weights.sum(axis=0) - 1).abs() < 1e-10).all()
    gradient_sum = gradients.sum(axis=0)
    assert ((gradient_sum & core.space).abs() < 1e-10) and ((gradient_sum & core.time).abs() < 1e-10)

    true_gradient = 1.3 * core.mv.x - 0.75 * core.mv.t
    corner_values = 0.4 + (true_gradient & corner_events)
    sample_values = 0.4 + (true_gradient & points)
    interpolated = (corner_values[:, None] * weights).sum(axis=0)
    reconstructed_gradient = (corner_values * gradients).sum(axis=0)
    assert ((interpolated - sample_values).abs() < 1e-10).all()
    gradient_residual = reconstructed_gradient - true_gradient
    assert ((gradient_residual & core.space).abs() < 1e-10) and ((gradient_residual & core.time).abs() < 1e-10)

    edge_forms = core.edge_whitney_forms(corners, points)
    tails, heads = core.triangle.corners[2, 1]
    classic_forms = (
        weights[tails] * gradients[heads, None] - weights[heads] * gradients[tails, None]
    ) * core.triangle.orientation[1].T
    form_residual = edge_forms - classic_forms
    assert ((form_residual & core.space).abs() < 1e-10).all()
    assert ((form_residual & core.time).abs() < 1e-10).all()

    sides = (core.triangle.boundary[2, 1] * (core.triangle.d[0] * corners)).batch()[:, 0]
    edge_events = corner_events[tails] * 0.5 + corner_events[heads] * 0.5
    readings = core.edge_whitney_forms(corners, edge_events) & sides[None, :]
    assert ((readings - np.eye(3)).abs() < 1e-10).all()

    right_null, left_null = core.mv.x - core.mv.t, core.mv.x + core.mv.t
    assert (right_null & core.Plane.dual()(right_null)).abs() < 1e-10
    assert (left_null & core.Plane.dual()(left_null)).abs() < 1e-10
    plane = 3 * core.mv.x + 4 * core.mv.t
    assert ((plane & core.Plane.dual()(plane)) + 7).abs() < 1e-10
    assert (-(plane & (core.mv.t >> core.Plane.dual()(plane))) - 25).abs() < 1e-10


def test_wake_and_moving_mirror_wave_run_and_draw():
    """Coarse wake and moving-mirror worldsheets solve and render both figures and one animation frame."""
    n_space, n_time = 5, 5
    corners, points = scenarios.single_triangle(core.space, core.time, 2)
    _, gradients, _ = core.corner_interpolation(corners, points)
    edge_forms = core.edge_whitney_forms(corners, points)
    wake_topology, wake_vertices, _ = scenarios.spacetime_strip(
        core.space, core.time, n_space, n_time, np.ones(n_time), scenarios.wake_duration,
    )
    _, wake_wave_operator, wake_areas = core.spacetime_operators(wake_topology, wake_vertices)
    source_event = scenarios.source_x * core.space + scenarios.source_t * core.time
    wake = core.causal_wake(
        wake_topology, wake_vertices, wake_wave_operator, wake_areas,
        n_space, n_time, source_event, scenarios.source_width,
    )
    plt.close(
        render.light_cone_figure(
            corners, points, gradients, edge_forms, wake_topology, wake_vertices, wake, source_event,
        )
    )

    n_time = 24
    times = np.linspace(0, scenarios.mirror_duration, n_time)
    wall_positions = scenarios.mirror_trajectory(
        times, scenarios.initial_length, scenarios.stroke, scenarios.inflection_time, scenarios.transition_time,
    )
    topology, vertices, dt = scenarios.spacetime_strip(
        core.space, core.time, n_space, n_time, wall_positions, scenarios.mirror_duration,
    )
    gradient, wave_operator, areas = core.spacetime_operators(topology, vertices)
    _, _, wave_rate, slice_energies = core.moving_mirror_wave(
        topology, vertices, gradient, wave_operator, areas, n_space, n_time, dt,
        scenarios.pulse_centre, scenarios.pulse_width, scenarios.wavelength,
    )
    plt.close(render.doppler_figure(topology, vertices, n_space, n_time, wave_rate, slice_energies, 1, 3))
    slices = core.cavity_slices(n_time, 1)
    frame = next(render.cavity_animation(topology, vertices, n_space, n_time, wave_rate, slices))
    assert frame.ndim == 3 and frame.shape[-1] == 3
