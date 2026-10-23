"""Slip in face-centred cubic crystals: slip directions in their planes, Schmid factors at most a half and one
over the root of six along [001], and pulls turning towards [112] on two systems."""

import matplotlib.pyplot as plt
import numpy as np

from numga import stack
from examples.mechanics.slip_systems import core, render, scenarios


def test_slip_directions_lie_in_their_planes_and_schmid_factors_are_at_most_a_half():
    directions, factors, planes, slips = scenarios.chart()
    assert np.abs((planes[:, None] & slips).to_array()).max() < 1e-12
    assert np.abs(factors.to_array()).max() <= 0.5 + 1e-12
    along = np.abs(core.schmid(core.mv.z.dual(), planes, slips).to_array())   # [planes, directions]
    assert np.isclose(along.max(), 1 / np.sqrt(6)) and np.isclose(along, along.max()).sum() == 8


def test_pulls_turn_to_the_double_slip_corner_on_two_systems_and_the_figures_draw():
    axes, amounts = scenarios.paths()                                         # [steps + 1, starts] Direction
    ends = (stack([core.mv.x, core.mv.y, core.mv.z]) & axes[-1][:, None]).to_array()
    assert (np.sort(ends, axis=-1) @ (np.array([1, 1, 2]) / np.sqrt(6)) > 0.999).all()
    carried = np.sort(np.abs(amounts.to_array()).sum(axis=0).reshape(len(scenarios.STARTS), -1), axis=-1)   # [starts, systems]
    assert (carried[:, -2:].sum(axis=-1) / carried.sum(axis=-1) > 0.95).all()
    directions, factors, planes, slips = scenarios.chart()
    plt.close(render.draw_chart(directions, factors, planes, slips, axes, amounts))
    assert len(list(render.animate_paths(directions, factors, axes, amounts, 2))) == 2
