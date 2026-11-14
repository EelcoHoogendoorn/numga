"""Three-point pose reconstruction is equivariant under rigid world-frame changes, and its static and animated views render."""

from itertools import islice

import matplotlib.pyplot as plt
import numpy as np

from examples.geometry.three_point_pose import core, render, scenarios


def test_reconstruction_is_equivariant_under_world_frame_motions():
    source, target, increments = scenarios.marker_scene()
    frame = (core.mv.yw * 0.8).exp() * (core.mv.xz * -0.7).exp()                   # [] Motor+
    moved_source, moved_target = frame >> source, frame >> target                  # [3] Point-, [3] Point-
    moved_increments = core.alignment_motors(moved_source, moved_target)           # [3] Motor+
    placed = list(core.alignments(source, increments, np.ones(1)))
    moved_placed = list(core.alignments(moved_source, moved_increments, np.ones(1)))
    assert ((placed[-1] & target).norm() < 1e-10).all()
    for stage, moved_stage in zip(placed, moved_placed):
        assert ((moved_stage & (frame >> stage)).norm() < 1e-10).all()


def test_pose_figure_and_animation_render():
    source, target, increments = scenarios.marker_scene()
    figure = render.pose_figure(source, target, source)
    plt.close(figure)
    states = core.alignments(source, increments, np.ones(1))
    frames = list(render.pose_animation(source, target, islice(states, 1)))
    assert len(frames) == 1
