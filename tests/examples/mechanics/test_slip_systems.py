"""Slip in face-centred cubic crystals: the chart on a couple of directions, the pulls slipped one step, and the
figures drawn."""

from functools import partial

from examples.mechanics.slip_systems import core, render, scenarios


def test_the_pulls_slip_a_step_and_the_figures_draw():
    _, _, planes, slips = scenarios.chart(2)
    axes, amounts = scenarios.paths(1)                                        # [2, starts] Direction-
    schmid = partial(core.schmid, planes=planes, directions=slips)
    render.draw_chart(schmid, planes, slips, axes, amounts)
    next(render.animate_paths(schmid, axes, amounts, 1))
