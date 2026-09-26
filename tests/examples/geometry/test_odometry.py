"""Odometry around a lap: each pose's uncertainty, from dead reckoning and from the information, is its block of the
inverse curvature of the objective; Gauss-Newton reaches the most likely poses in the plane and in space;
and the figures draw."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples import instantiate
from examples.geometry.odometry import render, scenarios


def inverse_curvature(core, poses, readings, weights, tails, heads, anchors, anchor_weights) -> np.ndarray:
    """Each pose's block of the inverse of the objective's curvature in the twists of every pose, by
    central differences of its gradient."""
    size, count = len(core.Twist.output_subspace), poses.shape[-1]
    columns = []
    for pose in range(count):
        for coefficient in range(size):
            nudge = np.zeros((count, size))
            nudge[pose, coefficient] = 1e-6
            step = (core.mv(core.Twist, nudge) * 0.5).exp()
            up = core.gradient(poses * step, readings, weights, tails, heads, anchors, anchor_weights).kernel.ravel()
            down = core.gradient(poses * step.inverse(), readings, weights, tails, heads, anchors, anchor_weights).kernel.ravel()
            columns.append((up - down) / 2e-6)
    inverse = np.linalg.inv(np.stack(columns, axis=1)).reshape(count, size, count, size)
    return inverse[np.arange(count), :, np.arange(count), :]


def test_uncertainties_are_the_inverse_curvature():
    steps = scenarios.lap_in_space()
    core = instantiate(scenarios.CORE, steps.algebra)
    truth, dead, readings, noises, tails, heads, anchors, priors = scenarios.survey(core, steps, 0)
    weights, anchor_weights = noises.inverse(), priors.inverse()

    # The step readings alone: dead reckoning's uncertainty, up to the other poses' vague priors.
    along = slice(0, len(steps))
    expected = inverse_curvature(core, dead, readings[along], weights[along], tails[along], heads[along], anchors, anchor_weights)
    np.testing.assert_allclose(core.reckon(dead, noises, priors).kernel, expected, atol=1e-5)

    # Every reading: at the most likely poses, the marginals are the Gauss-Newton ones; the full curvature
    # differs by what the remaining mismatches bend.
    *_, (poses, uncertainty) = scenarios.damped(steps, 1.0, 6, 0)[-1]
    expected = inverse_curvature(core, poses, readings, weights, tails, heads, anchors, anchor_weights)
    marginals = core.marginals(poses, noises, tails, heads, priors, weights, anchor_weights)
    np.testing.assert_allclose(marginals.kernel, expected, atol=1e-5)


def test_the_lap_closes_in_the_plane_and_draws():
    steps = scenarios.lap_in_plane()
    truth, dead, reckoned, poses, uncertainty = scenarios.closing(steps, 5, 0)
    for figure in (render.draw_lap(truth, dead, reckoned, dead[[0, -1]]),
                   render.draw(truth, dead, reckoned, poses, uncertainty)):
        assert isinstance(figure, plt.Figure)
        plt.close(figure)
    # Every pose's ellipse lies SIGMAS standard deviations out from the point it carries.
    here = poses >> render.ORIGIN
    quadric = render.ellipses(poses, uncertainty)
    np.testing.assert_allclose((quadric(here) & here).to_array(), -render.SIGMAS**2, atol=1e-6)
    # A frame for each damped iteration.
    *_, iterates = scenarios.damped(steps, 0.2, 2, 0)
    assert len(render.animate(truth, dead, reckoned, iterates)) == 2


def test_the_same_core_in_space():
    # A short lap in space that pitches and rolls reaches its most likely poses by the checks inside the
    # scenario. Its twists have six coefficients, where the plane's have three.
    steps = scenarios.lap_in_space()
    scenarios.closing(steps, 5, 0)
    assert len(instantiate(scenarios.CORE, steps.algebra).Twist.output_subspace) == 6
