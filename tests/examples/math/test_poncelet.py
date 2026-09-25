"""Poncelet's porism: paths of five sides about the conic touching an inscribed pentagon close from
every start on the ellipse, and about the conic of a shrunk pentagon they miss; the scenes pass these
checks and draw."""

from __future__ import annotations

import matplotlib.pyplot as plt

from examples.math.poncelet import render, scenarios


def test_scenes_pass_their_checks_and_draw():
    scenes = list(scenarios.turn(5))
    figure = render.draw(*scenes[0])
    assert isinstance(figure, plt.Figure)
    plt.close(figure)
    assert len(render.animate(scenes)) == 5
