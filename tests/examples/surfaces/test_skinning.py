"""Skinning a twisted cylinder: blended motors keep it rigid about the axis, blended maps collapse it."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.surfaces.skinning import render, scenarios
from examples.surfaces.skinning.core import cylinder, radius


def test_motor_blends_keep_the_radius_and_the_map_blend_collapses_it():
    motor_skin, slerp_skin, matrix_skin = scenarios.skinning()
    np.testing.assert_allclose(radius(motor_skin).to_array(), 1.0, atol=1e-12)
    np.testing.assert_allclose(radius(slerp_skin).to_array(), 1.0, atol=1e-12)
    w = cylinder(scenarios.N_RINGS, scenarios.N_AROUND)[1].to_array()
    np.testing.assert_allclose(radius(matrix_skin).to_array(), np.sqrt(1 - 2 * w * (1 - w) * (1 - np.cos(scenarios.TWIST))), atol=1e-12)


def test_figure_renders():
    figure = render.draw_skinning(*scenarios.skinning(), scenarios.N_RINGS, scenarios.N_AROUND)
    assert isinstance(figure, plt.Figure)
