"""Poncelet's porism: paths of five sides about the conic touching an inscribed pentagon close from
every start on the ellipse, and about the conic of a shrunk pentagon they miss."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.math.poncelet import render, scenarios


def test_paths_close_about_the_inscribed_pentagons_conic_and_miss_about_the_shrunk_ones():
    outer, inner, vertices = scenarios.run(np.linspace(0.3, 5.9, 7), 5)
    np.testing.assert_allclose((outer(vertices) & vertices).to_array(), 0.0, atol=1e-12)
    gap = np.abs((vertices[5] - vertices[0]).kernel).max(axis=-1)
    np.testing.assert_allclose(gap[0], 0.0, atol=1e-11)
    assert gap[1].min() > 1e-2


def test_scenes_draw():
    scenes = list(scenarios.turn(5))
    figure = render.draw(*scenes[0])
    assert isinstance(figure, plt.Figure)
    plt.close(figure)
    assert len(list(render.animate(scenes))) == 5
