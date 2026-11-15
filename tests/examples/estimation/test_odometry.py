"""Pose-graph odometry reaches the most likely trajectory in three dimensions with six-component twists and renders planar recovery cleanly."""

from __future__ import annotations

from functools import partial

import matplotlib.pyplot as plt

from examples import instantiate
from examples.estimation.odometry import render, scenarios
from numga.algebras import PGA2D


def test_spatial_lap_reaches_optimum():
    """Six-step spatial odometry with pitch and roll drives the covariance-weighted squared gradient below `1e-12` of dead reckoning."""
    steps = scenarios.spatial_lap_steps(scenarios.n_space_steps)
    space = instantiate(scenarios.core_module, steps.algebra)
    *_, poses, uncertainty = scenarios.closed_lap_solution(steps, scenarios.n_full_steps, 0)
    dead_gradient, optimum_gradient = scenarios.endpoint_gradients(steps, poses, 0)
    assert (uncertainty(optimum_gradient) & optimum_gradient).max() < 1e-12 * (uncertainty(dead_gradient) & dead_gradient).max()
    assert space.n_twist_modes == 6


def test_short_planar_lap_draws_and_animates():
    """A four-step planar lap closes in one step and renders its figures and one damped animation frame."""
    steps = scenarios.planar_lap_steps(4)
    plane = instantiate(scenarios.core_module, PGA2D)
    truth, dead_reckoning, reckoned_uncertainty, poses, uncertainty = scenarios.closed_lap_solution(steps, 1, scenarios.seed)
    reckoned_ellipsoids = plane.position_ellipsoids(dead_reckoning, reckoned_uncertainty, scenarios.sigmas)
    plt.close(render.dead_reckoning_figure(truth, dead_reckoning, reckoned_ellipsoids, dead_reckoning[[0, -1]]))
    plt.close(
        render.trajectory_figure(
            truth, dead_reckoning, reckoned_ellipsoids, poses, plane.position_ellipsoids(poses, uncertainty, scenarios.sigmas),
        )
    )
    _, _, _, states = scenarios.damped_closing_states(steps, scenarios.damping, 1, scenarios.seed)
    next(render.closing_animation(truth, dead_reckoning, reckoned_ellipsoids, partial(plane.position_ellipsoids, sigmas=scenarios.sigmas), states))
