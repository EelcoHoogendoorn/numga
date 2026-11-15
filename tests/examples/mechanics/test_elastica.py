"""Coverage tests step the elastica rods on two pieces, advance the telephone cord one step, and render every figure."""

import matplotlib.pyplot as plt

from numga import stack
from examples.mechanics.elastica import core, render, scenarios

n_pieces = 2


def test_elastica_rods_cord_step_and_figures() -> None:
    plt.close(render.draw_curls(scenarios.curled_strips(n_pieces) >> core.centre))
    list(render.animate_sweep(scenarios.euler_rods(scenarios.sweep_angles[:1], n_pieces) >> core.centre))
    coiled_rods = scenarios.kirchhoff_helices(n_pieces)                                             # [4, n_pieces + 1] Motor+
    plt.close(render.draw_helices(coiled_rods >> core.centre, scenarios.ribbon_edges(coiled_rods, scenarios.helix_ribbon_width)))
    uncoiled_rods = scenarios.uncoiling_wire(1, n_pieces)                                           # [1, n_pieces + 1] Motor+
    list(render.animate_uncoiling(uncoiled_rods >> core.centre, scenarios.ribbon_edges(uncoiled_rods, scenarios.uncoil_ribbon_width)))
    moved_cord = stack(list(scenarios.cord_frames(1, 1)))                                           # [1, 32] Motor+
    drawn_cord = core.dense_frames(moved_cord, scenarios.n_subdivisions)                            # [1, 125] Motor+
    list(render.animate_cord(drawn_cord >> core.centre, scenarios.ribbon_edges(drawn_cord, scenarios.cord_ribbon_width)))
