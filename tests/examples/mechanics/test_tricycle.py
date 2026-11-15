"""The planar and spatial tricycles each drive two steps and render one animation frame."""

from __future__ import annotations

from examples.mechanics.tricycle import render, scenarios

# Two steps provide the shortest rear-axle trail segment.
n_steps = 2


def test_the_plane_and_space_drive_a_step_and_draw():
    for core, draw in zip(scenarios.scenes.values(), (render.plane, render.space)):
        trajectory = scenarios.run(core, n_steps)
        list(render.animate(draw, scenarios.moments(*trajectory, 1)))
