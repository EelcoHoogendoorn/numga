"""Exact marker registration and the constraints preserved during each alignment."""

import numpy as np

from examples.geometry.three_point_pose import core, scenarios


def test_batched_marker_registration_and_intermediate_constraints():
    source, _, _ = scenarios.markers()
    angles = np.linspace(-0.9, 0.9, 17)
    poses = (core.mv.xw * angles + core.mv.zw * (angles / 3)).exp() * (
        core.mv.xy * angles + core.mv.yz * (angles / 2)).exp()
    target = poses[:, None] >> source
    increments = core.reconstruct(source, target)
    translated = increments[:, 0, None] >> source
    aligned = increments[:, 1, None] >> translated
    restored = increments[:, 2, None] >> aligned
    fractions = np.linspace(0, 1, 9)
    turns = (1 - fractions[:, None] + fractions[:, None] * increments[:, 1]).normalized()
    rolls = (1 - fractions[:, None] + fractions[:, None] * increments[:, 2]).normalized()

    # checks: the first point and then the whole first edge stay fixed along the paths.
    np.testing.assert_allclose((translated[:, 0] - target[:, 0]).kernel, 0, atol=1e-12)
    np.testing.assert_allclose(((turns >> translated[:, 0]) - target[:, 0]).kernel, 0, atol=1e-12)
    np.testing.assert_allclose((aligned[:, :2] - target[:, :2]).kernel, 0, atol=1e-12)
    np.testing.assert_allclose(((rolls[..., None] >> aligned[:, :2]) - target[:, :2]).kernel,
                               0, atol=1e-12)
    np.testing.assert_allclose((restored - target).kernel, 0, atol=1e-12)


def test_reconstruction_is_independent_of_the_world_frame():
    source, target, increments = scenarios.markers()
    frame = (core.mv.yw * 0.8).exp() * (core.mv.xz * -0.7).exp()
    moved_source, moved_target = frame >> source, frame >> target
    moved_increments = core.reconstruct(moved_source, moved_target)
    placed, moved_placed = source, moved_source
    for increment, moved_increment in zip(increments, moved_increments):
        placed = increment >> placed
        moved_placed = moved_increment >> moved_placed
        np.testing.assert_allclose((moved_placed - (frame >> placed)).kernel, 0, atol=1e-12)
