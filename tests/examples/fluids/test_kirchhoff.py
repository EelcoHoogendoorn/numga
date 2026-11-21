"""Kirchhoff's equations: added-fluid inertia from the confocal pencil matches Lamb's table, total inertia is symmetric and covariant under motors, symmetry-axis turns move no fluid, and one step draws."""

from __future__ import annotations

import numpy as np

from numga import dim, stack
from examples.fluids.kirchhoff import core, render, scenarios

mv = core.mv
x, y, z = core.axes                                                           # [] Plane+, Plane+, Plane+
density_unit = dim.mass / dim.length**3


def test_spheroid_matches_lambs_table():
    """For a prolate spheroid of aspect ratio 3, the fluid-to-solid kinetic energy ratios match Lamb's table."""
    body_stretch = core.ellipsoid_stretch(mv.scalar([[3.0], [1.0], [1.0]], unit=dim.length))   # [] Direction- <- Direction-, length
    shape = body_stretch(body_stretch)                                        # [] Direction- <- Direction-, length^2
    solid_inertia = core.total_inertia(shape, mv.scalar(1.0, unit=density_unit), mv.scalar(0.0, unit=density_unit), scenarios.n_widenings)
    fluid_inertia = core.total_inertia(shape, mv.scalar(0.0, unit=density_unit), mv.scalar(1.0, unit=density_unit), scenarios.n_widenings)
    rates = stack([mv.wx * (dim.length / dim.time), mv.wy * (dim.length / dim.time), mv.zx / dim.time])  # [3] Twist+, 1/time | length/time
    ratios_ext = (rates & fluid_inertia(rates)) / (rates & solid_inertia(rates))
    assert ratios_ext.unit is dim.none
    ratios = ratios_ext.to_array()
    np.testing.assert_allclose(ratios, [0.122, 0.803, 0.465], atol=1e-3)


def test_inertia_is_symmetric_and_moves_with_the_body():
    """Kinetic energy is a symmetric form on twists, and turning an ellipsoid by a motor conjugates its inertia."""
    body_stretch = core.ellipsoid_stretch(mv.scalar([[2.0], [1.3], [0.6]], unit=dim.length))   # [] Direction- <- Direction-, length
    rho_s, rho_f = mv.scalar(1.5, unit=density_unit), mv.scalar(1.0, unit=density_unit)
    inertia = core.total_inertia(body_stretch(body_stretch), rho_s, rho_f, scenarios.n_widenings)   # [] Forque- <- Twist+
    rng = np.random.default_rng(0)
    rate_type = core.Twist.derive.unit(1 / dim.time).derive.projective(dim.length)
    first_twist, second_twist = mv(rate_type, rng.normal(size=(2, 6)))        # [] Twist+, Twist+
    energy_12 = first_twist & inertia(second_twist)
    assert energy_12.unit == dim.mass * dim.length**2 / dim.time**2
    np.testing.assert_allclose(energy_12.to_array(), (second_twist & inertia(first_twist)).to_array(), rtol=1e-12)
    motor = (mv.xy * -0.4).exp() * (mv.yz * 0.7).exp()                        # [] Motor+
    turned_stretch = motor >> body_stretch(motor << core.Direction)           # [] Direction- <- Direction-, length
    turned_inertia = core.total_inertia(turned_stretch(turned_stretch), rho_s, rho_f, scenarios.n_widenings)   # [] Forque- <- Twist+
    np.testing.assert_allclose(((motor >> second_twist) & turned_inertia(motor >> first_twist)).to_array(),
                               (second_twist & inertia(first_twist)).to_array(), rtol=1e-10)


def test_turning_about_the_axis_of_symmetry_moves_no_fluid():
    """Turning a spheroid about its axis of symmetry displaces no fluid."""
    body_stretch = core.ellipsoid_stretch(mv.scalar([[0.3], [1.0], [1.0]], unit=dim.length))   # [] Direction- <- Direction-, length
    fluid_inertia = core.total_inertia(
        body_stretch(body_stretch), mv.scalar(0.0, unit=density_unit), mv.scalar(1.0, unit=density_unit), scenarios.n_widenings,
    )                                                                         # [] Forque- <- Twist+
    spin = mv.yz / dim.time
    assert ((spin & fluid_inertia(spin)).abs() < mv.scalar(1e-12, unit=dim.mass * dim.length**2 / dim.time**2)).all()


def test_the_bodies_glide_one_step_and_draw():
    _, impulse, _ = scenarios.launch_setup()
    body_outline, nose = scenarios.body_geometry()                            # [n_outline] Point-, [] Point-
    next(render.animate_gliding(impulse, body_outline, nose, scenarios.gliding_frames(1)))
