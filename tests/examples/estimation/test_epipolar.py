"""Coverage test runs one Gauss–Newton epipolar update, textbook matrix recovery, and one ray-traced frame."""

from __future__ import annotations

import numpy as np

from numga import stack
from examples.estimation.epipolar import core, render, scenarios, textbook


def test_one_step_reconstructs_and_draws() -> None:
    """One Gauss–Newton update, error evaluation, textbook baseline, and one rendered frame execute cleanly."""
    landmarks, images, _, rays, reconstructed_landmarks, true_motor, estimated_motors = scenarios.epipolar_scene(
        scenarios.sensor_noise, 1, scenarios.noise_seed,
    )
    _, _, initial_rms = scenarios.pose_and_landmark_errors(
        landmarks, reconstructed_landmarks[0], true_motor, estimated_motors[0],
    )
    axis_cosines, baseline_cosine, step_rms = scenarios.pose_and_landmark_errors(
        landmarks, reconstructed_landmarks[1], true_motor, estimated_motors[1],
    )
    assert (axis_cosines > 0.9).all()
    assert (baseline_cosine > 0.9).all()
    assert (step_rms < initial_rms).all()

    axes = stack((core.mv.x, core.mv.y, core.mv.z))
    ray_directions = (axes & images[..., None]).to_array()
    unit_rays = ray_directions / np.linalg.norm(ray_directions, axis=-1, keepdims=True)
    essential = textbook.essential_matrix(unit_rays[0], unit_rays[1])
    rotation, translation = textbook.candidate_pose(essential, unit_rays[0], unit_rays[1])
    textbook.ray_midpoints(np.zeros(3), unit_rays[0], -rotation.T @ translation, (rotation.T @ unit_rays[1].T).T)
    textbook.epipolar_lines(ray_directions[0], essential)

    states = scenarios.eased_states(rays, estimated_motors, 1, 1)
    frame = next(render.reconstruction_animation(landmarks, scenarios.house_edges, true_motor, states))
    assert frame.shape[1:] == (render.n_pixels, 3)
