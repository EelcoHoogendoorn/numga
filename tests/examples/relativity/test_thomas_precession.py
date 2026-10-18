"""Thomas precession and BMT spin dynamics: the scenes pass their checks and draw."""

from __future__ import annotations

import matplotlib.pyplot as plt

from examples.relativity.thomas_precession import render, scenarios


def test_the_scenes_pass_their_checks_and_draw():
    loop_pos, loop_u, _, loop_lab = scenarios.thomas_loop()
    ring_pos, _, _, ring_lab = scenarios.storage_ring()
    figure = render.draw_thomas_summary(loop_pos, loop_u, loop_lab, ring_pos, ring_lab)
    plt.close(figure)
    assert len(render.animate_storage_ring(ring_pos[:4], ring_lab[:4], 6)) == 1
