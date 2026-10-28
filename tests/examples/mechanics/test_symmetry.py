"""The cube's rotations, the averaged responses' invariants, and the figures."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.mechanics.symmetry import core, render, scenarios
from examples.mechanics.symmetry.core import mv


def test_cube_rotations_are_24_distinct_rotations():
    """Every pair of the 24 rotors differs, even up to sign: their scalar overlap is not plus or minus one."""
    cube = core.cube_rotations()
    overlap = abs((~cube[:, None]).scalar_product(cube[None, :]).to_array())
    assert cube.shape == (24,)
    assert ((overlap > 1 - 1e-9).sum(axis=1) == 1).all()


def test_the_cube_makes_conduction_isotropic_and_removes_rotary_power_with_the_inversion():
    _, surfaces, _, _ = scenarios.heat_conduction()
    assert (surfaces[-1] - core.sphere * scenarios.gains.mean()).norm().to_array().max() < 1e-7
    _, rotary_surfaces = scenarios.optical_activity()
    assert rotary_surfaces[5].norm().to_array().max() < 1e-12
    mean_power = mv(core.RotaryPower, [scenarios.rotary_values.mean()])
    assert (rotary_surfaces[4] - mean_power).norm().to_array().max() < 1e-12


def test_figures_draw():
    responses, _, driving, fluxes = scenarios.heat_conduction()
    centre, axial, diagonal, corners, stiffness, conductivity, _ = scenarios.crystal_lattice()
    rotary, _ = scenarios.optical_activity()
    figures = [render.draw_conduction(responses, driving, fluxes),
               render.draw_crystal(centre, axial, diagonal, corners, conductivity, stiffness),
               render.draw_optical_activity(rotary)]
    assert all(isinstance(figure, plt.Figure) for figure in figures)
    for figure in figures:
        plt.close(figure)
