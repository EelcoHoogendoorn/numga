"""Kirchhoff's equations: the fluid's inertia from the confocal family matches Lamb's coefficients,
turning a body about its axis of symmetry moves no fluid, and the scenes draw."""

from __future__ import annotations

from itertools import islice

import matplotlib.pyplot as plt
import numpy as np

from numga import stack
from examples.fluids.kirchhoff import core, render, scenarios

mv = core.mv
x, y, z = core.axes


def test_spheroid_matches_lambs_table():
    """For a prolate spheroid of aspect three the fluid's kinetic energy over the solid's is 0.122 shifting
    along its axis, 0.803 across it, and 0.465 turning it, in Lamb's table, to its three digits. The table
    is laid out by eccentricity, and lists this spheroid's aspect rounded to 2.99."""
    stretch = core.stretch(np.array([3, 1, 1]))
    shape = stretch(stretch)
    solid = core.inertia(shape, 1, 0, scenarios.N_WIDENINGS)
    fluid = core.inertia(shape, 0, 1, scenarios.N_WIDENINGS)
    rates = stack([mv.w ^ x, mv.w ^ y, z ^ x])
    ratios = ((rates & fluid(rates)) / (rates & solid(rates))).to_array()
    np.testing.assert_allclose(ratios, [0.122, 0.803, 0.465], atol=1e-3)


def test_inertia_is_symmetric_and_moves_with_the_body():
    """The kinetic energy is a symmetric form on rates, for an ellipsoid with three different axes, and
    an ellipsoid turned by a motor has the turned inertia: the confocal family needs no principal frame."""
    stretch = core.stretch(np.array([2.0, 1.3, 0.6]))
    inertia = core.inertia(stretch(stretch), 1.5, 1, scenarios.N_WIDENINGS)
    rng = np.random.default_rng(0)
    first, second = mv.bivector(rng.normal(size=(2, 6)))
    np.testing.assert_allclose((first & inertia(second)).to_array(), (second & inertia(first)).to_array(), rtol=1e-12)
    motor = ((x ^ y) * -0.4).exp() * ((y ^ z) * 0.7).exp()
    turned = motor >> stretch(motor << core.Direction)
    moved = core.inertia(turned(turned), 1.5, 1, scenarios.N_WIDENINGS)
    np.testing.assert_allclose(((motor >> second) & moved(motor >> first)).to_array(), (second & inertia(first)).to_array(), rtol=1e-10)


def test_turning_about_the_axis_of_symmetry_moves_no_fluid():
    """A spheroid turned about its axis of symmetry stays where it was, so the fluid stays at rest."""
    stretch = core.stretch(np.array([0.3, 1, 1]))
    fluid = core.inertia(stretch(stretch), 0, 1, scenarios.N_WIDENINGS)
    np.testing.assert_allclose(((y ^ z) & fluid(y ^ z)).to_array(), 0, atol=1e-12)


def test_scenes_draw():
    figure = render.draw_coefficients(scenarios.ASPECTS, scenarios.coefficients())
    assert isinstance(figure, plt.Figure)
    plt.close(figure)
    motors = scenarios.gliding()
    _, impulse, _ = scenarios.launch()
    assert len(render.animate_gliding(impulse, islice(scenarios.frames(motors), 2))) == 2
