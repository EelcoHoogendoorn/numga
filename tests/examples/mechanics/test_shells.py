"""Discrete shells: the disc and the strip on a handful of vertices relax for a frame of one step, and the scenes
draw."""

from __future__ import annotations

from examples.mechanics.shells import render, scenarios

# The vertices' spacing on the disc, a handful across it, and on the strip, the widest that keeps three rows across it.
disc_spacing, strip_spacing = 0.5, 0.15


def test_the_sheets_relax_a_step_and_the_scenes_draw():
    disc, disc_frames = scenarios.disc(disc_spacing, 1, 1)
    strip, strip_frames = scenarios.strip(strip_spacing, 1, 1)
    render.draw_settled([(disc.topology.elements[2], disc_frames[-1]), (strip.topology.elements[2], strip_frames[-1])])
    list(render.animate(strip.topology.elements[2], strip_frames[-1:]))
