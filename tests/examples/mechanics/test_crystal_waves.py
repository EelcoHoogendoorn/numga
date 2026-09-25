"""Elastic waves in cubic crystals: the stiffness reads back its elastic constants, energy runs
along the wave only in an isotropic material, and the figures render."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.mechanics.crystal_waves import core, render, scenarios


def test_stiffness_reads_back_the_elastic_constants():
    """The traction's components C11, C12 and C44 on the cube's faces, from the strain modes and
    their moduli: a stretch along x pulls on the x face with C11 and on the y face with C12, and a
    shear of the xy face pulls on it with C44."""
    x, y = core.axes[0], core.axes[1]
    for name, (c11, c12, c44, _) in scenarios.MATERIALS.items():
        stiffness, _ = scenarios.crystal(name)
        read = [x | stiffness(x, x, x), y | stiffness(y, x, x), x | stiffness(y, x, y)]
        np.testing.assert_allclose([value.to_array() for value in read], [c11, c12, c44], rtol=1e-10)


def test_energy_follows_the_wave_only_when_isotropic():
    """In fused silica every wave carries its energy along its heading and the two shear waves share
    one speed; in silicon the shear energy swings away from the heading."""
    heading = core.mv.vector(np.random.default_rng(1).normal(size=(64, 3))).normalized()
    for name, isotropic in (("fused silica", True), ("silicon", False)):
        stiffness, density = scenarios.crystal(name)
        values, polarization = core.waves(stiffness, heading)
        velocity = core.energy_flow(stiffness, heading, polarization, density)
        along = (velocity | heading[:, None]).to_array()
        speed = (velocity | velocity).square_root().to_array()
        swing = np.arccos(np.clip(along / speed, -1, 1)).max()                # the largest angle off the heading
        if isotropic:
            assert swing < 1e-5
            np.testing.assert_allclose(values[:, 0].to_array(), values[:, 1].to_array(), rtol=1e-8)
        else:
            assert swing > 0.2


def test_figures_render():
    """The scenarios pass their checks, and both figures draw."""
    names = list(scenarios.MATERIALS)
    for figure in (render.draw_focusing([scenarios.focusing(name, 2000, 0) for name in names], names, 32),
                   render.draw_wave_fronts([scenarios.wave_fronts(name, 64) for name in names], names)):
        assert isinstance(figure, plt.Figure)
        plt.close(figure)
