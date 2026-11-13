"""The elastica's rods on a couple of pieces each, the telephone cord a frame of one step, and every figure drawn."""

from numga import stack

from examples.mechanics.elastica import core, render, scenarios

# The pieces along each rod: the fewest that bend it.
n_pieces = 2


def test_the_rods_take_their_shapes_the_cord_steps_and_the_figures_draw():
    render.draw_curls(scenarios.curling(n_pieces) >> core.centre)
    list(render.animate_sweep(scenarios.euler(scenarios.SWEEP[:1], n_pieces) >> core.centre))
    coiled = scenarios.kirchhoff(n_pieces)                                    # [tilts, pieces + 1] Motor+
    render.draw_helices(coiled >> core.centre, scenarios.ribbon(coiled, scenarios.RIBBON))
    uncoiled = scenarios.uncoiling(1, n_pieces)                               # [1, pieces + 1] Motor+
    list(render.animate_uncoiling(uncoiled >> core.centre, scenarios.ribbon(uncoiled, scenarios.UNCOIL_RIBBON)))
    moved = stack(list(scenarios.pulled(1, 1)))                               # [1, pieces + 2] Motor+
    cord = core.between(moved, scenarios.N_DRAWN_BETWEEN)                     # [1, drawn] Motor+
    list(render.animate_cord(cord >> core.centre, scenarios.ribbon(cord, scenarios.CORD_RIBBON)))
