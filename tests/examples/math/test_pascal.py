"""Pascal's theorem: the conic through five points passes through them and every sixth point found on
it, the crossings of the hexagon's opposite sides lie on one line, and Pascal's condition on the sixth
point is the conic; the scenes pass these checks and draw."""

from __future__ import annotations

import matplotlib.pyplot as plt

from examples.math.pascal import render, scenarios


def test_scenes_pass_their_checks_and_draw():
    scenes = list(scenarios.turn(6))
    figure = render.draw(*scenes[0])
    assert isinstance(figure, plt.Figure)
    plt.close(figure)
    assert len(render.animate(scenes)) == 6
