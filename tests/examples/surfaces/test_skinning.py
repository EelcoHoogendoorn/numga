"""Blended bone motors keep a twisted cylinder at unit radius about its axis, whereas blended point maps collapse it toward the chord, and the figure renders."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.surfaces.skinning import render, scenarios
from examples.surfaces.skinning.core import cylinder, mv


def test_motor_blends_keep_the_radius_and_the_map_blend_collapses_it():
    n_rings, n_around = 3, 6
    vertices, weights = cylinder(n_rings, n_around)                            # [n_rings * n_around] Point-, [n_rings * n_around]
    motor_skin, slerp_skin, matrix_skin = scenarios.skinning(vertices, weights, scenarios.twist_angle)
    assert (((mv.yz & motor_skin.normalized()).norm() - 1).abs() < 1e-12).all()
    assert (((mv.yz & slerp_skin.normalized()).norm() - 1).abs() < 1e-12).all()
    chord_radius = np.sqrt(1 - 2 * weights * (1 - weights) * (1 - np.cos(scenarios.twist_angle)))
    assert (((mv.yz & matrix_skin.normalized()).norm() - chord_radius).abs() < 1e-12).all()
    plt.close(render.draw_skinning(motor_skin, slerp_skin, matrix_skin, n_rings, n_around))
