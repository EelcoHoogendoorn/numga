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
    along_x, along_y = x.dual(), y.dual()
    stiffness, _ = scenarios.crystals()
    read = [stiffness(x, along_x, x) & along_x, stiffness(y, along_x, x) & along_y, stiffness(y, along_x, y) & along_x]
    for value, constant in zip(read, (scenarios.C11, scenarios.C12, scenarios.C44)):
        np.testing.assert_allclose(value.to_array(), constant, rtol=1e-10)


def test_energy_follows_the_wave_only_when_isotropic():
    """In fused silica every wave carries its energy along its heading and the two shear waves share
    one speed; in silicon the shear energy swings away from the heading."""
    heading = core.mv.vector(np.random.default_rng(1).normal(size=(64, 3))).normalized()
    stiffness, density = scenarios.crystals()
    values, polarization = core.waves(stiffness[:, None], heading)
    velocity = core.energy_flow(stiffness[:, None], heading, polarization, density[:, None])
    along = (heading[:, None] & velocity).to_array()
    speed = velocity.dot(velocity).square_root().to_array()
    swing = np.arccos(np.clip(along / speed, -1, 1)).max(axis=(1, 2))         # the largest angle off the heading
    silica, silicon = scenarios.NAMES.index("fused silica"), scenarios.NAMES.index("silicon")
    assert swing[silica] < 1e-5 and swing[silicon] > 0.2
    np.testing.assert_allclose(values[silica, :, 0].to_array(), values[silica, :, 1].to_array(), rtol=1e-8)


def test_figures_render():
    """Along its heading the group velocity has the phase speed, the speeds along a cube edge and a
    face diagonal are the textbook ones, and both figures draw."""
    names = scenarios.NAMES
    stiffness, density = scenarios.crystals()
    velocity, heading, values = scenarios.focusing(2000, 0)
    phase = (values / density[:, None, None]).square_root()
    np.testing.assert_allclose((heading[:, None] & velocity).to_array(), phase.to_array(), rtol=1e-8)
    edge_and_diagonal = core.waves(stiffness[:, None], core.mv.vector([[1.0, 0, 0], [1, 1, 0]]).normalized())[0]
    speeds = (edge_and_diagonal / density[:, None, None]).square_root().to_array()
    C11, C12, C44 = scenarios.C11, scenarios.C12, scenarios.C44
    np.testing.assert_allclose(speeds[:, 0], np.sqrt(np.stack([C44, C44, C11], axis=-1) / density[:, None]), rtol=1e-8)
    expected = np.sort(np.stack([(C11 - C12) / 2, C44, (C11 + C12 + 2 * C44) / 2], axis=-1), axis=-1)
    np.testing.assert_allclose(speeds[:, 1], np.sqrt(expected / density[:, None]), rtol=1e-8)
    for figure in (render.draw_focusing(velocity, names, 32),
                   render.draw_wave_fronts(scenarios.wave_fronts(64), names)):
        assert isinstance(figure, plt.Figure)
        plt.close(figure)
