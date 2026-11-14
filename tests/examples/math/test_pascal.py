"""Pascal's theorem: the quadric through five points vanishes on them and on every sixth point along
turning lines, opposite hexagon sides meet in three collinear crossings, and Pascal's bilinear form
matches the quadric up to scale."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.math.pascal import core, render, scenarios
from numga import stack


def test_crossings_are_collinear_and_pascal_form_matches_quadric():
    five = scenarios.ellipse_points(np.array([0.2, 1.9, 2.8, 4.0, 5.5]))
    shape = core.quadric(five)
    states = list(core.hexagon_configurations(five, np.array([0.3, 1.4, 2.5])))
    hexagons = stack([state[0] for state in states])
    crossings = stack([state[2] for state in states])
    joins = crossings[:, 0] & crossings[:, 1] & crossings[:, 2]
    sizes = crossings[:, 0].norm() * crossings[:, 1].norm() * crossings[:, 2].norm()
    assert ((shape(five) & five).abs() < 1e-11).all()
    assert ((shape(hexagons) & hexagons).abs() < 1e-11).all()
    assert ((joins / sizes).abs() < 1e-10).all()
    probes = core.mv.point(np.random.default_rng(1).normal(size=(5, 2)))
    ratios = core.pascal_form(five)(probes, probes) / (shape(probes) & probes)
    assert ((ratios - ratios[0]).abs() < 1e-11 * ratios[0].abs()).all()


def test_one_frame_draws_and_animates():
    five = scenarios.ellipse_points(scenarios.five_angles)
    shape = core.quadric(five)
    state = next(core.hexagon_configurations(five, scenarios.turning_angles(1)))
    figure = render.draw(shape, *state)
    plt.close(figure)
    next(render.animate(shape, [state]))
