"""Fluxons in a planar Josephson junction on coarse grids: the meeting's first frame of pendula, the boosted fluxons
over one step, drawn, and the collapsing ring's and the reflected line's first frames."""

import matplotlib.pyplot as plt

from examples.electromagnetism.josephson import render, scenarios


def test_the_junctions_run_and_draw():
    next(render.animate_pendula(*scenarios.meeting(n_along=41)))
    plt.close(render.draw_contraction(*scenarios.contracting(n_along=41, n_running=1)))
    next(render.animate(*scenarios.collapsing(n_square=9)))
    _, _, currents = scenarios.reflecting(n_wide=9, n_high=5)
    next(currents)
