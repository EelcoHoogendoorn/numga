"""Pascal's theorem: the conic through five points passes through them and every sixth point found on
it, the crossings of the hexagon's opposite sides lie on one line, and Pascal's condition on the sixth
point is the conic."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.math.pascal import core, render, scenarios


def test_the_crossings_of_opposite_sides_lie_on_one_line_and_pascal_is_the_conic():
    five = scenarios.ellipse_points(np.array([0.2, 1.9, 2.8, 4.0, 5.5]))
    shape, hexagon, crossing = scenarios.run(five, np.linspace(0.1, 3.0, 7))
    np.testing.assert_allclose((shape(hexagon) & hexagon).to_array(), 0.0, atol=1e-11)
    join = (crossing[..., 0] & crossing[..., 1] & crossing[..., 2]).to_array()
    size = np.prod(np.linalg.norm(crossing.kernel, axis=-1), axis=-1)
    np.testing.assert_allclose(join / size, 0.0, atol=1e-10)
    probes = core.mv.point(np.random.default_rng(1).normal(size=(5, 2)))
    ratio = (core.pascal(five)(probes, probes) / (shape(probes) & probes)).to_array()
    np.testing.assert_allclose(ratio, ratio[0], rtol=1e-11)


def test_scenes_draw():
    scenes = list(scenarios.turn(6))
    figure = render.draw(*scenes[0])
    assert isinstance(figure, plt.Figure)
    plt.close(figure)
    assert len(list(render.animate(scenes))) == 6
