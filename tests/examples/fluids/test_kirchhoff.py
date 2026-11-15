"""Kirchhoff's equations: added-fluid inertia from the confocal pencil matches Lamb's table, total inertia is symmetric and covariant under motors, symmetry-axis turns move no fluid, and one step draws."""

from __future__ import annotations

import numpy as np

from numga import stack
from examples.fluids.kirchhoff import core, render, scenarios

mv = core.mv
x, y, z = core.axes                                                           # [] Plane+, Plane+, Plane+


def test_spheroid_matches_lambs_table():
    """For a prolate spheroid of aspect ratio 3, the fluid-to-solid kinetic energy ratios match Lamb's table."""
    body_stretch = core.ellipsoid_stretch(np.array([3, 1, 1]))                # [] Direction- <- Direction-
    shape = body_stretch(body_stretch)                                        # [] Direction- <- Direction-
    solid_inertia = core.total_inertia(shape, 1, 0, scenarios.n_widenings)    # [] Forque- <- Twist+
    fluid_inertia = core.total_inertia(shape, 0, 1, scenarios.n_widenings)    # [] Forque- <- Twist+
    rates = stack([mv.wx, mv.wy, mv.zx])                                      # [3] Twist+
    ratios = ((rates & fluid_inertia(rates)) / (rates & solid_inertia(rates))).to_array()
    np.testing.assert_allclose(ratios, [0.122, 0.803, 0.465], atol=1e-3)


def test_inertia_is_symmetric_and_moves_with_the_body():
    """Kinetic energy is a symmetric form on twists, and turning an ellipsoid by a motor conjugates its inertia."""
    body_stretch = core.ellipsoid_stretch(np.array([2.0, 1.3, 0.6]))          # [] Direction- <- Direction-
    inertia = core.total_inertia(body_stretch(body_stretch), 1.5, 1, scenarios.n_widenings)   # [] Forque- <- Twist+
    rng = np.random.default_rng(0)
    first_twist, second_twist = mv.bivector(rng.normal(size=(2, 6)))          # [] Twist+, Twist+
    np.testing.assert_allclose((first_twist & inertia(second_twist)).to_array(),
                               (second_twist & inertia(first_twist)).to_array(), rtol=1e-12)
    motor = (mv.xy * -0.4).exp() * (mv.yz * 0.7).exp()                        # [] Motor+
    turned_stretch = motor >> body_stretch(motor << core.Direction)           # [] Direction- <- Direction-
    turned_inertia = core.total_inertia(turned_stretch(turned_stretch), 1.5, 1, scenarios.n_widenings)   # [] Forque- <- Twist+
    np.testing.assert_allclose(((motor >> second_twist) & turned_inertia(motor >> first_twist)).to_array(),
                               (second_twist & inertia(first_twist)).to_array(), rtol=1e-10)


def test_turning_about_the_axis_of_symmetry_moves_no_fluid():
    """Turning a spheroid about its axis of symmetry displaces no fluid."""
    body_stretch = core.ellipsoid_stretch(np.array([0.3, 1, 1]))              # [] Direction- <- Direction-
    fluid_inertia = core.total_inertia(body_stretch(body_stretch), 0, 1, scenarios.n_widenings)   # [] Forque- <- Twist+
    assert ((mv.yz & fluid_inertia(mv.yz)).abs() < 1e-12).all()


def test_the_bodies_glide_one_step_and_draw():
    _, impulse, _ = scenarios.launch_setup()
    body_outline, nose = scenarios.body_geometry()                            # [n_outline] Point-, [] Point-
    next(render.animate_gliding(impulse, body_outline, nose, scenarios.gliding_frames(1)))
