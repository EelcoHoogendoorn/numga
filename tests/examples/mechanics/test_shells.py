"""Discrete shells: the disc and strip shells on coarse triangulations relax for one step and
render."""

from __future__ import annotations

import matplotlib.pyplot as plt

from examples.mechanics.shells import render, scenarios

disc_spacing, strip_spacing = 0.5, 0.15


def test_shells_relax_one_step_and_render():
    disc_topology, *_, disc_run = scenarios.disc(disc_spacing, 1, 1)
    disc_frames = list(disc_run)
    strip_topology, *_, strip_run = scenarios.strip(strip_spacing, 1, 1)
    strip_frames = list(strip_run)
    figure = render.draw_settled([
        (disc_topology.elements[2], disc_frames[-1]),
        (strip_topology.elements[2], strip_frames[-1]),
    ])
    plt.close(figure)
    list(render.animate(strip_topology.elements[2], strip_frames[-1:]))
