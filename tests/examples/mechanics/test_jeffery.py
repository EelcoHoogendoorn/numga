"""Jeffery orbits: the spin solve agrees with Jeffery's formula in the body's frame, a needle's axis
moves as a line of fluid, and the scenes draw."""

from __future__ import annotations

from itertools import islice

import matplotlib.pyplot as plt
import numpy as np

from examples.mechanics.jeffery import core, render, scenarios

x, y, z = core.axes


def test_spin_is_jefferys_formula_in_the_principal_frame():
    """For an ellipsoid along the axes, the spin in each plane is the fluid's, plus the strain rate in
    that plane times the difference over the sum of the squared semi-axes."""
    semi_axes = np.array([3.0, 1.5, 0.5])
    entries = np.array([[0.3, -1.2, 0.4], [0.7, 0.1, -0.5], [-0.2, 0.9, -0.4]])   # x_i & gradient(x_j.dual())
    gradient = (core.axes.dual()[:, None] * entries * (core.axes & core.Point)[None, :]).sum(axis=(0, 1))
    spin = core.spin(core.placed(core.stretch(semi_axes), core.mv.rotor()), gradient)
    fluid = core.curl(gradient) / 2
    strain = (entries + entries.T) / 2
    squares = semi_axes**2
    for first, second in [(0, 1), (1, 2), (0, 2)]:
        plane = core.axes[first] ^ core.axes[second]
        relative = (spin - fluid).dot(plane).to_array()
        expected = (squares[first] - squares[second]) / (squares[first] + squares[second]) * strain[first, second]
        np.testing.assert_allclose(relative, expected, atol=1e-12)


def test_needle_axis_moves_with_the_fluid():
    """The axis of a very long needle turns as a line of fluid does: its velocity is the gradient on it,
    less its own stretching."""
    gradient = x.dual() * (y & core.Point) + 0.3 * z.dual() * (x & core.Point)
    turn = ((x ^ z) * -0.4).exp() * ((x ^ y) * 0.3).exp()
    spin = core.spin(core.placed(core.stretch(np.array([1e4, 1.0, 1.0])), turn), gradient)
    axis = turn >> x.dual()
    along = gradient(axis)
    np.testing.assert_allclose(axis.commutator(spin).kernel, (along - axis * axis.dot(along)).kernel, atol=1e-6)


def test_scenes_draw():
    rotors = scenarios.tumbling()
    figure = render.draw_orbits(rotors >> x.dual())
    assert isinstance(figure, plt.Figure)
    plt.close(figure)
    heights = y.dual() * scenarios.HEIGHTS
    first = islice(scenarios.frames(rotors), 1)
    assert len(render.animate_tumbling(core.stretch(scenarios.SEMI_AXES), heights, scenarios.flow()(heights), first)) == 1
