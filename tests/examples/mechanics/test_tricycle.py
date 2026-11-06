"""The tricycle drives two steps in the plane and in space, and a frame of each draws."""

from __future__ import annotations

from examples.mechanics.tricycle import render, scenarios

# The steps of the drive: the fewest a trail of the rear axle's middle needs.
n_steps = 2


def test_the_plane_and_space_drive_a_step_and_draw():
    for core, draw in zip(scenarios.SCENES.values(), (render.plane, render.space)):
        drive = scenarios.run(core, n_steps)
        list(render.animate(draw, scenarios.moments(*drive, 1, n_steps)))
