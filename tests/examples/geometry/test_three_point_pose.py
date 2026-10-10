"""Exact marker registration and the constraints preserved during each alignment."""

import numpy as np
from numga import stack

from examples.geometry.three_point_pose import core, scenarios


def test_batched_marker_registration_and_intermediate_constraints():
    source, _, _ = scenarios.markers()
    angles = np.linspace(-0.9, 0.9, 17)
    poses = (core.mv.xw * angles + core.mv.zw * (angles / 3)).exp() * (
        core.mv.xy * angles + core.mv.yz * (angles / 2)).exp()
    target = poses[:, None] >> source
    increments = core.reconstruct(source, target)
    fractions = np.linspace(0, 1, 9)[1:]
    _, *paths = core.alignments(source, increments[..., None], fractions)
    translating, turning, rolling = stack(paths).reshape((len(increments), len(fractions)) + target.shape)

    # checks: the first point and then the whole first edge stay fixed along the paths.
    np.testing.assert_allclose((translating[-1, :, 0] - target[:, 0]).kernel, 0, atol=1e-10)
    np.testing.assert_allclose((turning[..., 0] - target[:, 0]).kernel, 0, atol=1e-10)
    np.testing.assert_allclose((rolling[..., :2] - target[:, :2]).kernel, 0, atol=1e-10)
    np.testing.assert_allclose((rolling[-1] - target).kernel, 0, atol=1e-10)


def test_reconstruction_is_independent_of_the_world_frame():
    source, target, increments = scenarios.markers()
    frame = (core.mv.yw * 0.8).exp() * (core.mv.xz * -0.7).exp()
    moved_source, moved_target = frame >> source, frame >> target
    moved_increments = core.reconstruct(moved_source, moved_target)
    for placed, moved_placed in zip(core.placements(source, increments),
                                    core.placements(moved_source, moved_increments)):
        np.testing.assert_allclose((moved_placed - (frame >> placed)).kernel, 0, atol=1e-12)
