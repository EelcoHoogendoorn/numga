"""Pose-graph odometry: a short lap in space that pitches and rolls reaches its most likely poses, with twists of six
coefficients, and a short lap in the plane closes by one step and draws."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from numga.algebras import PGA2D
from examples import instantiate
from examples.estimation.odometry import render, scenarios


def test_the_same_core_in_space():
    # A short lap in space that pitches and rolls reaches its most likely poses, where the gradient falls
    # a millionfold. Its twists have six coefficients, where the plane's have three.
    steps = scenarios.lap_in_space()
    *_, poses, _ = scenarios.closing(steps, 3, 0)
    dead_gradient, optimum_gradient = scenarios.gradients(steps, poses, 0)
    np.testing.assert_allclose(optimum_gradient.kernel, 0.0, atol=1e-6 * np.abs(dead_gradient.kernel).max())
    assert len(instantiate(scenarios.CORE, steps.algebra).Twist.output_subspace) == 6


def test_a_short_lap_closes_and_draws():
    steps = scenarios.lap_in_plane(n_steps=4)
    truth, dead_reckoning, reckoned_uncertainty, poses, uncertainty = scenarios.closing(steps, 1, scenarios.SEED)
    plane = instantiate(scenarios.CORE, PGA2D)
    plt.close(render.draw(truth, dead_reckoning, plane.ellipsoids(dead_reckoning, reckoned_uncertainty, scenarios.SIGMAS),
                          poses, plane.ellipsoids(poses, uncertainty, scenarios.SIGMAS)))
