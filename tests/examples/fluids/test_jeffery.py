"""Jeffery orbits: principal-frame spin formula, slender-needle material line limit, and one-step rendering."""

from __future__ import annotations

import numpy as np

from examples.fluids.jeffery import core, render, scenarios

x, y, z = core.axes


def test_spin_matches_principal_frame_formula() -> None:
    """For an axis-aligned ellipsoid, each plane's relative spin equals the strain rate scaled by the Bretherton ratio."""
    semi_axes = np.array([3.0, 1.5, 0.5])
    entries = np.array([[0.3, -1.2, 0.4], [0.7, 0.1, -0.5], [-0.2, 0.9, -0.4]])
    gradient = (core.axes.dual()[:, None] * entries * (core.axes & core.Point)[None, :]).sum(axis=(0, 1))
    spin = core.jeffery_spin(core.body_shapes(core.body_stretches(semi_axes), core.mv.rotor()), gradient)
    fluid_spin = core.flow_curl(gradient) / 2
    strain = (entries + entries.T) / 2
    squares = semi_axes**2
    for first, second in [(0, 1), (1, 2), (0, 2)]:
        plane = core.axes[first] ^ core.axes[second]
        relative = (spin - fluid_spin).scalar_product(plane.reverse())
        expected = (squares[first] - squares[second]) / (squares[first] + squares[second]) * strain[first, second]
        assert ((relative - expected).abs() < 1e-12).all()


def test_needle_axis_turns_with_material_line() -> None:
    """A slender needle's long axis turns at the ambient velocity gradient minus its radial stretching."""
    gradient = x.dual() * (y & core.Point) + 0.3 * z.dual() * (x & core.Point)
    rotor = ((x ^ z) * -0.4).exp() * ((x ^ y) * 0.3).exp()
    spin = core.jeffery_spin(core.body_shapes(core.body_stretches(np.array([1e4, 1.0, 1.0])), rotor), gradient)
    axis = rotor >> x.dual()
    material_velocity = gradient(axis)
    projected_velocity = material_velocity - axis * axis.scalar_product(material_velocity.reverse())
    assert ((axis.commutator(spin) - projected_velocity).norm() < 1e-6).all()


def test_one_step_tumble_and_draw() -> None:
    n_pixels = 48
    heights = y.dual() * scenarios.flow_heights                              # [6] Point-
    states = scenarios.shown_start_states(scenarios.orbit_stream(n_steps=2))
    next(render.animate_tumbling(core.body_stretches(scenarios.semi_axes), heights, scenarios.flow_gradient(heights),
                                 n_pixels, states))

