"""The stressed cube: the isotropic stress is Lamé's formula, and the scenes draw."""

from __future__ import annotations

from itertools import islice

import matplotlib.pyplot as plt
import numpy as np

from examples.mechanics.stress import core, render, scenarios


def test_isotropic_stress_is_lames_formula():
    """Entry (i, j) of the stress, x_i & stress(x_j), is lame * tr(strain) * delta_ij + 2 * mu * strain_ij."""
    lame, shear_modulus = 1.8, 1.1
    matrix = np.array([[0.02, 0.015, -0.005], [0.015, -0.01, 0.008], [-0.005, 0.008, 0.025]])
    strain = (core.axes[:, None] * matrix * (core.axes & core.Point)[None, :]).sum(axis=(0, 1))   # [] Strain
    stress = core.cauchy_stress(strain, lame, shear_modulus)
    entries = (core.axes[:, None] & stress(core.axes)[None, :]).to_array()
    np.testing.assert_allclose(entries, lame * np.trace(matrix) * np.eye(3) + 2 * shear_modulus * matrix, rtol=1e-12)


def test_scenes_draw():
    figure = render.draw_frames(*scenarios.frames())
    assert isinstance(figure, plt.Figure)
    plt.close(figure)
    _, values = scenarios.frames()
    assert len(render.animate_turning(values, islice(scenarios.turning(), 1))) == 1
