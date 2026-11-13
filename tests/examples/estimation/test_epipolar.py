"""Two-view reconstruction of the house under sensor noise, at one Gauss-Newton step, and the first frame of its search
drawn."""

from __future__ import annotations

import numpy as np

from examples.estimation.epipolar import render, scenarios


def test_one_step_reconstructs_and_draws():
    landmarks, _, _, rays, reconstructed_landmarks, true_motor, estimated_motors = scenarios.epipolar(scenarios.NOISE, 1)
    scenarios.errors(landmarks, reconstructed_landmarks[-1], true_motor, estimated_motors[-1])
    states = scenarios.easing(rays, estimated_motors, 1, 1)
    next(render.animate_reconstruction(landmarks, np.array(scenarios.HOUSE_EDGES), true_motor, states))
