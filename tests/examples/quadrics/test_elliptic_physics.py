"""Tests for quadric rigid-body physics on the 2-sphere and 3-sphere."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from numga import stack
from examples.quadrics.elliptic_physics import render, scenarios
from examples.quadrics.elliptic_physics.scenarios import S2, S3


def test_ellipse_tangency():
    """Great circles tangent to an ellipse satisfy `tangents & dual(tangents) == 0`."""
    half_widths = np.tan(np.radians([30.0, 15.0]))
    dual = S2.ellipsoid_quadric(half_widths)
    azimuths = np.radians([0.0, 30.0, 75.0, 120.0, 200.0, 310.0])
    turns = (S2.mv.xy * (-azimuths / 2)).exp()
    directions = turns >> S2.mv.x
    tangents = (
        S2.mv.x * ((directions | S2.mv.x) / half_widths[0])
        + S2.mv.y * ((directions | S2.mv.y) / half_widths[1])
        + S2.mv.z
    )
    assert ((tangents & dual(tangents)).abs() < 1e-12).all()


def test_overlap_margin_sign():
    """Overlap margin is positive when apart, near zero at tangency, and negative on overlap."""
    first = S2.ellipsoid_quadric(np.tan(np.radians([20.0, 20.0]))).inverse()
    angles = np.radians([55.0, 45.0, 35.0])
    turns = (S2.mv.xz * (-angles / 2)).exp()
    second_body = S2.ellipsoid_quadric(np.tan(np.radians([25.0, 25.0]))).inverse()
    second = S2.world_quadrics(turns, second_body)
    margin, _ = S2.quadric_overlap(first, second)
    assert margin[0] > 0.005
    assert margin[1].abs() < 2e-3
    assert margin[2] < -0.005


def test_ellipse_inertia_has_an_intermediate_axis():
    """A 35-by-10-degree ellipse of unit mass has three distinct principal moments `Iz < Ix < Iy`."""
    points, masses = scenarios.ellipse_samples(np.radians([[35.0, 10.0]]), np.array([1.0]), 384, 10)
    inertia = S2.cloud_inertia(points, masses)[0]
    axes = stack((S2.mv.yz, S2.mv.zx, S2.mv.xy))
    moments = (axes & inertia(axes)).abs()
    assert bool(moments[2] < moments[0]) and bool(moments[0] < moments[1])
    expected = S2.mv.scalar(np.array([0.913, 0.992, 0.094])[:, None])
    assert ((moments - expected).abs() < 0.03).all()


def test_collision_conserves_energy_and_momentum():
    """One elastic impulse between two overlapping ellipses conserves kinetic energy and world momentum."""
    placements = (S2.mv.yz * (-np.radians([0.0, 28.0]) / 2)).exp()
    _, motors, momenta, duals, forms, inverse_inertias, _ = scenarios.ellipse_population(
        np.array([[25.0, 15.0], [25.0, 15.0]]),
        np.array([1.0, 1.0]),
        placements,
        np.array([[-2.5, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        ["#38bdf8", "#f43f5e"],
        96,
    )
    motors = scenarios.camera.inverse() * motors
    energy_before = S2.kinetic_energy(inverse_inertias(momenta), momenta)
    momentum_before = S2.world_momentum(motors, momenta)
    momenta_after, applied = S2.elastic_collision(
        motors, momenta, forms, duals, inverse_inertias, np.array([0]), np.array([1]), 0.001,
    )
    assert applied == 1
    energy_after = S2.kinetic_energy(inverse_inertias(momenta_after), momenta_after)
    momentum_after = S2.world_momentum(motors, momenta_after)
    assert (((energy_after - energy_before).abs() / energy_before.abs()) < 1e-12).all()
    assert (((momentum_after - momentum_before).antinorm() / momentum_before.antinorm()) < 1e-7).all()


def test_interior_samples_lie_inside_with_the_given_mass():
    """Interior points fill both an ellipsoid and a Clifford torus and sum to the requested mass."""
    rng = np.random.default_rng(0)
    for dual in (
        S3.ellipsoid_quadric(np.array([[0.3, 0.2, 0.1]])),
        S3.diagonal_quadric(np.array([[-1.0, -1.0, 2.0, 3.0]])),
    ):
        points, masses = S3.interior_samples(dual, np.array([2.0]), 64, rng)
        form = dual.inverse()
        assert ((points & form[:, None](points)) < 0).all()
        assert ((masses.sum(axis=-1) - 2).abs() < 1e-12).all()


def test_a_scene_on_each_sphere_steps_and_draws():
    """Stepping one frame on the 2-sphere and on the 3-sphere exercises all hemisphere, chart, and ray-traced renderers."""
    pixels = render.hemisphere_points()
    inside_fig = render.inside_figure(((S2.mv.z & pixels) > 0.8))
    plt.close(inside_fig)

    colors, motors, momenta, duals, forms, inverse_inertias, bounding_radii = scenarios.tumbling_scene()
    states = list(
        S2.collision_frames(motors, momenta, forms, duals, inverse_inertias, bounding_radii, 2, scenarios.s2_dt, 1)
    )
    frame = next(render.s2_animation(forms, colors, 32, states[:1]))
    assert frame.shape == (32, 32, 3)
    contact_fig = render.contact_figure(S2.world_quadrics(motors, forms), colors)
    plt.close(contact_fig)
    rates = stack([inverse_inertias(state[1])[0] for state in states])
    tumbling_fig = render.tumbling_figure(rates, scenarios.s2_dt)
    plt.close(tumbling_fig)

    s3_colors, s3_motors, s3_momenta, s3_duals, s3_forms, s3_inverse, s3_radii, eye, light = scenarios.s3_crowd_scene()
    s3_states = S3.collision_frames(
        s3_motors, s3_momenta, s3_forms, s3_duals, s3_inverse, s3_radii, 1, scenarios.s3_dt, 1,
    )
    s3_image = next(render.s3_animation(s3_forms, s3_colors, eye, light, (24, 32), s3_states))
    assert s3_image.shape == (24, 32, 3)
    frame_fig = render.frame_figure(s3_image)
    plt.close(frame_fig)
