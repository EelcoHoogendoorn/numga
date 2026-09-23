"""The symmetry scenarios check their invariants; each figure draws."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pytest

from examples.mechanics.symmetry import core, render, scenarios


def test_cube_rotations_are_24_distinct_rotations():
    """Every pair of the 24 rotors differs, even up to sign: their scalar overlap is not ±1."""
    cube = core.cube_rotations()
    overlap = abs((~cube[:, None] * cube[None, :]).select[0].to_array())
    assert cube.shape == (24,)
    assert ((overlap > 1 - 1e-9).sum(axis=1) == 1).all()


@pytest.mark.parametrize("scenario, draw", [
    (scenarios.heat_conduction, render.draw_conduction),
    (scenarios.flywheel, render.draw_flywheel),
    (scenarios.crystal_lattice, render.draw_crystal),
])
def test_scenario_checks_and_figure(scenario, draw):
    assert isinstance(draw(*scenario()), plt.Figure)


def test_mathematics_does_not_import_plotting():
    import subprocess
    import sys

    probe = (
        "import examples.mechanics.symmetry.core, sys; "
        "print([m for m in sys.modules if m.split('.')[0] in ('matplotlib', 'PIL')])"
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]", out.stdout
