"""Blended bone motors keep a twisted cylinder at unit radius about its axis, whereas blended point maps collapse it toward the chord."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.surfaces.skinning import render, scenarios
from examples.surfaces.skinning.core import cylinder, mv


def test_motor_blends_keep_the_radius_and_the_map_blend_collapses_it():
    motor_skin, slerp_skin, matrix_skin = scenarios.skinning()
    _, weights = cylinder(scenarios.N_RINGS, scenarios.N_AROUND)
    np.testing.assert_allclose((motor_skin.normalized() & mv.yz).norm().to_array(), 1.0, atol=1e-12)
    np.testing.assert_allclose((slerp_skin.normalized() & mv.yz).norm().to_array(), 1.0, atol=1e-12)
    chord_radius = np.sqrt(1 - 2 * weights * (1 - weights) * (1 - np.cos(scenarios.TWIST)))
    np.testing.assert_allclose((matrix_skin.normalized() & mv.yz).norm().to_array(), chord_radius, atol=1e-12)


def test_figure_renders():
    figure = render.draw_skinning(*scenarios.skinning(), scenarios.N_RINGS, scenarios.N_AROUND)
    assert isinstance(figure, plt.Figure)
    plt.close(figure)
