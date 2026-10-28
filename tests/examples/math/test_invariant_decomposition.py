"""A six-dimensional rotation: its bivector splits into the same three commuting planes by its wedge
powers and by the spectrum of the acceleration of a turned direction; a turned point stays on a circle in
each plane; a rigid motion splits into three placed planes; and the orbit draws."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.math.invariant_decomposition import core, render, scenarios


def test_both_decompositions_agree_and_the_orbit_draws():
    bivector, start, motion = scenarios.samples(1)
    parts, points, circles = scenarios.tangle(bivector, start, 60)
    squares, _ = core.from_spectrum(bivector, core.Antivector)
    wedge_squares, wedge_parts = core.from_wedge_powers(bivector)
    np.testing.assert_allclose((parts.sum(axis=0) - bivector).kernel, 0.0, atol=1e-11)
    np.testing.assert_allclose((wedge_squares - squares).kernel, 0.0, atol=1e-11)
    np.testing.assert_allclose((wedge_parts - parts).kernel, 0.0, atol=1e-10)
    np.testing.assert_allclose((core.rotor(squares, parts, core.mv.scalar([1])) - (bivector / 2).exp()).kernel, 0.0, atol=1e-6)
    radii = circles.scalar_norm_squared()
    np.testing.assert_allclose((radii - radii[:, :1]).kernel, 0.0, atol=1e-6)
    planes = core.placed(motion)
    np.testing.assert_allclose((motion - planes.sum(axis=-1)).kernel, 0.0, atol=1e-10)
    figure = render.draw(points, circles, parts, 30)
    assert isinstance(figure, plt.Figure)
    plt.close(figure)
    assert len(list(render.animate(points, circles, parts, 3))) == 3
