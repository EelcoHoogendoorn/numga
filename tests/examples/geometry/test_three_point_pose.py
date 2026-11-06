"""Exact marker registration is independent of the world frame, and its first alignment step draws."""

from itertools import islice

import numpy as np

from examples.geometry.three_point_pose import core, render, scenarios


def test_reconstruction_is_independent_of_the_world_frame():
    source, target, increments = scenarios.markers()
    frame = (core.mv.yw * 0.8).exp() * (core.mv.xz * -0.7).exp()
    moved_source, moved_target = frame >> source, frame >> target
    moved_increments = core.reconstruct(moved_source, moved_target)
    for placed, moved_placed in zip(core.placements(source, increments),
                                    core.placements(moved_source, moved_increments)):
        np.testing.assert_allclose((moved_placed - (frame >> placed)).kernel, 0, atol=1e-12)


def test_the_first_alignment_step_draws():
    source, target, increments = scenarios.markers()
    states = core.alignments(source, increments, np.ones(1))
    list(render.animate(islice(states, 2), source, target))
