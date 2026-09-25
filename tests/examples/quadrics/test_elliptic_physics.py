"""Tests for quadric rigid-body physics on S² and S³, one engine instantiated for each sphere."""

from __future__ import annotations

from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np

from examples.quadrics.elliptic_physics import render, scenarios
from examples.quadrics.elliptic_physics.scenarios import S2, S3, ellipse_mesh


def test_ellipse_tangency():
    """Great circles tangent to an ellipse satisfy tangents & Q(tangents) == 0."""
    th_x, th_y = np.radians(30.0), np.radians(15.0)
    Q = S2.ellipsoid(np.tan(np.array([th_x, th_y])))
    phi = np.radians([0.0, 30.0, 75.0, 120.0, 200.0, 310.0])
    tangents = S2.mv.x * (np.cos(phi) / np.tan(th_x)) + S2.mv.y * (np.sin(phi) / np.tan(th_y)) + S2.mv.z
    np.testing.assert_allclose((tangents & Q(tangents)).to_array(), 0.0, atol=1e-12)


def test_overlap_margin_sign():
    """The margin is positive when apart, near zero when touching, negative when overlapping."""
    C1 = S2.ellipsoid(np.tan(np.radians([20.0, 20.0]))).inverse()
    # Centres apart, touching at 45°, the sum of the radii 20° and 25°, and overlapping.
    alphas = np.radians([55.0, 45.0, 35.0])
    turn = (S2.mv.xz * (-alphas / 2.0)).exp()
    C2 = turn >> S2.ellipsoid(np.tan(np.radians([25.0, 25.0]))).inverse()(turn << S2.Point)
    margin = S2.overlap(C1, C2)[0].to_array()
    assert margin[0] > 0.005
    np.testing.assert_allclose(margin[1], 0.0, atol=2e-3)
    assert margin[2] < -0.005


def test_ellipse_inertia_has_an_intermediate_axis():
    """A 35° by 10° ellipse of unit mass has three distinct moments, Iz < Ix < Iy."""
    points, masses = ellipse_mesh(np.radians(np.array([[35.0, 10.0]])), np.array([1.0]), 1536, 10)
    inertia = S2.pointcloud_inertia(points, masses)[0]
    axes = [S2.mv.yz, S2.mv.zx, S2.mv.xy]
    Ix, Iy, Iz = (abs((axis & inertia(axis)).to_array()) for axis in axes)
    assert Iz < Ix < Iy
    np.testing.assert_allclose([Iz, Ix, Iy], [0.094, 0.913, 0.992], atol=0.03)


def test_free_flight_conserves_energy_and_momentum():
    """The Lie midpoint step keeps the body-frame momentum's norm exactly and the energy closely."""
    points, masses = ellipse_mesh(np.radians(np.array([[35.0, 10.0]])), np.array([1.0]), 384, 10)
    inertia = S2.pointcloud_inertia(points, masses)[0]
    I_inv = inertia.inverse()
    motor, momentum = S2.mv.rotor(), inertia(S2.mv.yz * 2.5 + S2.mv.xy * 0.05)
    energy = (I_inv(momentum) & momentum).to_array()
    size = momentum.norm().to_array()
    for _ in range(200):
        motor, momentum = S2.step_motor(motor, momentum, I_inv, 0.01)
    np.testing.assert_allclose(momentum.norm().to_array(), size, atol=1e-14)
    np.testing.assert_allclose((I_inv(momentum) & momentum).to_array(), energy, rtol=0.01)


def test_collision_conserves_energy_and_momentum():
    """Two overlapping, approaching ellipses: one elastic impulse keeps total energy and world momentum."""
    # Centres 28° apart, overlapping.
    placement = (S2.mv.yz * (-np.radians([0.0, 28.0]) / 2.0)).exp()
    bodies = scenarios.ellipses(np.array([[25.0, 15.0], [25.0, 15.0]]), np.array([1.0, 1.0]), placement,
                            np.array([[-2.5, 0.0, 0.0], [2.0, 0.0, 0.0]]), ["#38bdf8", "#f43f5e"], 96)
    bodies = replace(bodies, motor=scenarios.CAMERA.inverse() * bodies.motor)
    after, applied = S2.collide(bodies, np.array([0]), np.array([1]), 0.001)
    assert applied == 1
    np.testing.assert_allclose(after.kinetic_energy().to_array(), bodies.kinetic_energy().to_array(), rtol=1e-12)
    change = (after.total_momentum() - bodies.total_momentum()).norm() / bodies.total_momentum().norm()
    assert change.to_array() < 1e-7


def test_filled_points_lie_inside_with_the_given_mass():
    """The sampler's points fill the inside of any quadric, an ellipsoid or a torus, and carry its mass."""
    rng = np.random.default_rng(0)
    for Q in (S3.ellipsoid(np.array([[0.3, 0.2, 0.1]])), S3.quadric(np.array([[-1.0, -1.0, 2.0, 3.0]]))):
        points, masses = S3.filled(Q, np.array([2.0]), 200, rng)
        C = Q.inverse()
        assert ((points & C[:, None](points)) < 0.0).all()
        np.testing.assert_allclose(masses.sum(axis=-1).to_array(), 2.0)


def test_s2_scenarios_render():
    """Each S² scene conserves its invariants (its checks) and renders a figure and an animation."""
    for scene, steps, draw, impulses in (
        (scenarios.crowded, 41, render.draw_collisions, 5),
        (scenarios.hyperbolic, 41, render.draw_collisions, 15),
        (scenarios.tumbling, 100, render.draw_tumbling, 0),
    ):
        trajectory, colors = scene(steps)
        assert trajectory.impulses == impulses
        assert isinstance(draw(trajectory, colors, 0.015), plt.Figure)
        frames = render.hemisphere_frames(trajectory.surfaces[::10], colors, 80, 1)
        assert frames and frames[0].ndim == 3 and all(f.shape == frames[0].shape for f in frames)


def test_s3_scenarios_render():
    """Each S³ scene collides, conserves energy and keeps the tunnel's wall intact (its checks), and renders."""
    for scene in (scenarios.crowd, scenarios.gap, scenarios.needle, scenarios.tunnel):
        trajectory, colors, eye, light = scene(12)
        frames = render.s3_frames(trajectory, colors, eye, light, (45, 60), 1)
        assert isinstance(render.draw_last_frame(frames, colors.shape[0]), plt.Figure)
        assert frames and frames[0].ndim == 3 and all(f.shape == frames[0].shape for f in frames)
