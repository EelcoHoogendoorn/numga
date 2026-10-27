"""Tests for pose-graph odometry verify that dead-reckoned and marginal covariances match the blocks of the inverse curvature, that Gauss-Newton converges in the plane and in space, and that the figures render."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from numga.algebras import PGA2D
from examples import instantiate
from examples.estimation.odometry import render, scenarios


def inverse_curvature(core, poses, readings, weights, tails, heads, anchors, anchor_weights) -> np.ndarray:
    """Compute each pose's diagonal block of the inverse curvature of the objective by central differences of its gradient."""
    n_modes, n_poses = len(core.Twist.output_subspace), poses.shape[-1]
    columns = []
    for pose in range(n_poses):
        for coefficient in range(n_modes):
            nudge = np.zeros((n_poses, n_modes))
            nudge[pose, coefficient] = 1e-6
            step = (core.mv(core.Twist, nudge) * 0.5).exp()
            up_gradient = core.gradient(poses * step, readings, weights, tails, heads, anchors, anchor_weights).kernel.ravel()
            down_gradient = core.gradient(poses * step.inverse(), readings, weights, tails, heads, anchors, anchor_weights).kernel.ravel()
            columns.append((up_gradient - down_gradient) / 2e-6)
    inverse_hessian = np.linalg.inv(np.stack(columns, axis=1)).reshape(n_poses, n_modes, n_poses, n_modes)
    return inverse_hessian[np.arange(n_poses), :, np.arange(n_poses), :]


def test_uncertainties_are_the_inverse_curvature():
    steps = scenarios.lap_in_space()
    core = instantiate(scenarios.CORE, steps.algebra)
    truth, dead_reckoning, readings, noises, tails, heads, anchors, priors = scenarios.survey(core, steps, 0)
    weights, anchor_weights = noises.inverse(), priors.inverse()

    # The step readings alone give dead reckoning's uncertainty, up to the remaining poses' vague priors.
    step_slice = slice(0, len(steps))
    expected = inverse_curvature(
        core, dead_reckoning, readings[step_slice], weights[step_slice], tails[step_slice], heads[step_slice], anchors, anchor_weights
    )
    np.testing.assert_allclose(core.reckon(dead_reckoning, noises, priors).kernel, expected, atol=1e-5)

    # Every reading: at the most likely poses, the marginals match the Gauss-Newton inverse curvature up to the
    # residual curvature of the remaining mismatches.
    *_, poses = core.gauss_newton(dead_reckoning, readings, weights, tails, heads, anchors, anchor_weights, 1.0, 3)
    expected = inverse_curvature(core, poses, readings, weights, tails, heads, anchors, anchor_weights)
    marginals = core.marginals(poses, noises, tails, heads, priors, weights, anchor_weights)
    np.testing.assert_allclose(marginals.kernel, expected, atol=1e-5)


def test_the_lap_closes_in_the_plane_and_draws():
    steps = scenarios.lap_in_plane()
    truth, dead_reckoning, reckoned_uncertainty, poses, uncertainty = scenarios.closing(steps, 3, 0)
    plane = instantiate(scenarios.CORE, PGA2D)
    reckoned_ellipses = plane.ellipses(dead_reckoning, reckoned_uncertainty, scenarios.SIGMAS)
    quadric = plane.ellipses(poses, uncertainty, scenarios.SIGMAS)
    for figure in (render.draw_lap(truth, dead_reckoning, reckoned_ellipses, dead_reckoning[[0, -1]]),
                   render.draw(truth, dead_reckoning, reckoned_ellipses, poses, quadric)):
        assert isinstance(figure, plt.Figure)
        plt.close(figure)
    # Every pose's ellipse lies SIGMAS standard deviations out from the point it carries.
    positions = poses >> plane.origin
    np.testing.assert_allclose((quadric(positions) & positions).to_array(), -scenarios.SIGMAS**2, atol=1e-6)
    # A frame for each damped iteration.
    *_, iterates = scenarios.damped(steps, 0.2, 2, 0)
    frames = render.animate(truth, dead_reckoning, reckoned_ellipses,
                            ((poses, plane.ellipses(poses, uncertainty, scenarios.SIGMAS)) for poses, uncertainty in iterates))
    assert len(list(frames)) == 2


def test_the_same_core_in_space():
    # A short lap in space that pitches and rolls reaches its most likely poses, where the gradient falls
    # a millionfold. Its twists have six coefficients, where the plane's have three.
    steps = scenarios.lap_in_space()
    *_, poses, _ = scenarios.closing(steps, 3, 0)
    dead_gradient, optimum_gradient = scenarios.gradients(steps, poses, 0)
    np.testing.assert_allclose(optimum_gradient.kernel, 0.0, atol=1e-6 * np.abs(dead_gradient.kernel).max())
    assert len(instantiate(scenarios.CORE, steps.algebra).Twist.output_subspace) == 6
