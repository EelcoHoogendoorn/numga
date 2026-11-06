"""Elastic waves in cubic crystals: the stiffness reads back its elastic constants, energy runs
along the wave only in an isotropic material, and the figures render on a handful of headings."""

from __future__ import annotations

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
    velocity, _, _ = scenarios.focusing(8, scenarios.SEED)
    render.draw_focusing(velocity, scenarios.NAMES)
    render.draw_wave_fronts(scenarios.wave_fronts(8), scenarios.NAMES)
