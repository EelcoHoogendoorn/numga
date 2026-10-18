"""Low-Reynolds-number swimming and Stokeslets in PGA2D: the scene passes its checks and draws."""

from __future__ import annotations

import matplotlib.pyplot as plt

from examples.mechanics.stokes_swimmer import render, scenarios


def test_the_scene_passes_its_checks_and_draws():
    angles, _, positions, world_sites, world_forces, grid, flow = scenarios.strokes()
    figure = render.draw_summary(angles, positions, world_sites, world_forces, grid, flow[0])
    plt.close(figure)
    assert len(render.animate_swimmers(positions[:4], world_sites[:3], grid, flow[:1], 3)) == 1
