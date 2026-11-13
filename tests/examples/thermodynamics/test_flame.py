"""Tests for a cellular flame check energy conservation along the reaction direction and step and render one frame on a
coarse triangulated disc."""

import numpy as np

from examples.thermodynamics.flame import core, render, scenarios


def test_reaction_conserves_enthalpy():
    np.testing.assert_allclose((core.energy & core.reaction).to_array(), 0.0, atol=1e-15)


def test_burning_runs_and_draws():
    topology, vertices, frames = scenarios.flame_states(4, scenarios.n_unshown + 1, 1)
    next(render.animate(topology, vertices, scenarios.radius, core.energy, frames))
