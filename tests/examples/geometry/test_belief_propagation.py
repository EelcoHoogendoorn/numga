"""The pose graph: on a chain the beliefs' covariances are the exact ones, on the closed loop the poses
settle where the gradient vanishes, and the survey draws."""

from __future__ import annotations

import matplotlib.pyplot as plt

from examples.geometry.belief_propagation import render, scenarios


def test_scenarios_pass_their_checks_and_draw():
    alone = scenarios.chain(400, 0)
    closed = scenarios.loop(400, 0)
    runs = {"odometry alone": alone, "loop closed": closed[:-1]}
    truth, dead = alone[:2]
    for figure in (render.draw_lap(truth, dead, dead[[0, -1]]), render.draw_survey(runs)):
        assert isinstance(figure, plt.Figure)
        plt.close(figure)
    assert len(render.animate_survey(runs, 3)) == 3
