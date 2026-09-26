"""The pose graph: on a chain the beliefs' covariances are the exact ones, on the closed loop the poses
settle where the gradient vanishes, and the survey draws; the same core in the plane and in space."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples import instantiate
from examples.geometry.belief_propagation import render, scenarios


def test_scenarios_pass_their_checks_and_draw():
    steps = scenarios.lap_in_plane()
    alone = scenarios.chain(steps, 400, 0)
    closed = scenarios.loop(steps, scenarios.PLANE_LAP, 400, 0)
    runs = {"odometry alone": alone, "loop closed": closed[:-1]}
    truth, dead = alone[:2]
    for figure in (render.draw_lap(truth, dead, dead[[0, -1]]), render.draw_survey(runs)):
        assert isinstance(figure, plt.Figure)
        plt.close(figure)
    assert len(render.animate_survey(runs, np.arange(3))) == 3
    # The rollout grows the lap a pose at a time, then shows every pose after the loop is closed.
    truth, dead, poses, quadrics = closed[:-1]
    assert len(render.animate_growth(truth, dead, scenarios.growing(poses, quadrics, 2, np.arange(1)))) == 3


def test_the_same_core_in_space():
    # A short lap in space that pitches and rolls: the chain's beliefs are exact and the closed loop
    # settles, by the checks inside the scenarios. Its twists have six coefficients, where the plane's
    # have three.
    steps = scenarios.lap_in_space()
    core = instantiate(scenarios.CORE, steps.algebra)
    _, _, poses, quadrics = scenarios.chain(steps, 10, 0)
    scenarios.loop(steps, scenarios.SPACE_LAP, 100, 0)
    # Rolled out along the chain first, then closed and carried on from what was told: it settles too.
    scenarios.closing(steps, scenarios.SPACE_LAP, scenarios.SPACE_POSES, 100, 0)
    assert len(core.Twist.output_subspace) == 6
    # Every pose's quadric, an ellipsoid, lies SIGMAS standard deviations out from the point it carries.
    here = poses[-1] >> scenarios.origin(core)
    np.testing.assert_allclose((quadrics[-1](here) & here).to_array(), -scenarios.SIGMAS**2, atol=1e-6)
