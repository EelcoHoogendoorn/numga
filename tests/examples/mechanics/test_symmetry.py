"""The symmetry scenarios check their invariants; each figure draws."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pytest

from examples.mechanics.symmetry import core, render, scenarios


def test_cube_rotations_are_24_distinct_rotations():
    """Every pair of the 24 rotors differs, even up to sign: their scalar overlap is not plus or minus one."""
    cube = core.cube_rotations()
    overlap = abs((~cube[:, None]).scalar_product(cube[None, :]).to_array())
    assert cube.shape == (24,)
    assert ((overlap > 1 - 1e-9).sum(axis=1) == 1).all()


@pytest.mark.parametrize("scenario, draw", [
    (scenarios.heat_conduction, render.draw_conduction),
    (scenarios.flywheel, render.draw_flywheel),
    (scenarios.crystal_lattice, render.draw_crystal),
])
def test_scenario_checks_and_figure(scenario, draw):
    assert isinstance(draw(*scenario()), plt.Figure)
