"""Coverage tests for material symmetry: cube rotations, group-averaged invariants, and GPU figure builders."""

from __future__ import annotations

import matplotlib.pyplot as plt

from examples.mechanics.symmetry import core, render, scenarios
from examples.mechanics.symmetry.core import mv


def test_cube_rotations_are_24_distinct_rotations():
    """Every pair among the 24 cube rotors differs even up to sign."""
    cube = core.cube_rotations()                                                # [24] Rotor+
    overlap = (~cube[:, None]).scalar_product(cube[None, :]).abs()              # [24, 24] Scalar+
    assert cube.shape == (24,)
    assert ((overlap > 1 - 1e-9).sum(axis=1) == 1).all()


def test_group_averages_enforce_neumann_invariants():
    """Cube rotations make conduction isotropic, lattice stiffness stays cubic, and spatial inversion cancels rotary power."""
    responses, surfaces, _, _ = scenarios.heat_conduction()
    assert ((surfaces[-1] - scenarios.sphere_directions * scenarios.gains.mean()).norm() < 1e-7).all()
    assert (((mv.x & responses[2](mv.x)) - (mv.y & responses[2](mv.y))).abs() < 1e-6).all()

    _, _, _, _, _, conductivity, elasticity = scenarios.crystal_lattice()
    assert (((scenarios.unit_fields & conductivity(scenarios.unit_fields)) - 2 / 3).abs() < 1e-7).all()
    octahedral_face = (mv.x + mv.y + mv.z).normalized()
    assert ((elasticity(mv.x, mv.x, mv.x, mv.x) - 1 / 2).abs() < 1e-7).all()
    assert ((elasticity(octahedral_face, octahedral_face, octahedral_face, octahedral_face) - 1 / 3).abs() < 1e-7).all()

    _, rotary_surfaces = scenarios.optical_activity()
    assert (rotary_surfaces[5].norm() < 1e-9).all()
    mean_power = mv(core.RotaryPower, [scenarios.rotary_values.mean()])
    assert ((rotary_surfaces[4] - mean_power).norm() < 1e-9).all()


def test_figures_render():
    """All three symmetry figure builders execute and return matplotlib figures without saving files."""
    responses, _, driving, fluxes = scenarios.heat_conduction()
    centre, axial, diagonal, corners, stiffness_surface, conductivity, _ = scenarios.crystal_lattice()
    gyrations, _ = scenarios.optical_activity()
    for fig in (
        render.conduction_figure(responses[:2], driving, fluxes[:2]),
        render.crystal_figure(centre, axial, diagonal, corners, conductivity, stiffness_surface),
        render.optical_activity_figure(gyrations[:2]),
    ):
        assert fig.axes
        plt.close(fig)
