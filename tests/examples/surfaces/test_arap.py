"""As-rigid-as-possible deformation: the bending scene runs on a coarse bar for one frame of one iteration, and
the frame draws."""

from __future__ import annotations

from examples.surfaces.arap import render, scenarios


def test_the_bending_scene_runs_and_draws():
    topology, rest_vertices, rigid_frames, *_ = scenarios.bending(n_divisions=2, n_frames=1, n_iterations=1)
    assert len(list(render.animate(topology, rest_vertices, rigid_frames, scenarios.CELL))) == 1
