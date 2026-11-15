"""Freezing a seed of ice in undercooled water conserves energy, and the seed grows for a step on a coarse dish and
renders its first frame."""
import numpy as np

from examples.thermodynamics.snowflake import core, render, scenarios

mv = core.mv


def test_freezing_keeps_the_energy():
    """Freezing moves energy from the latent store into heat, so the energy plane pairs with the freezing direction to zero."""
    freezing = core.ice + scenarios.latent_heat * core.heat
    energy = mv.h - scenarios.latent_heat * mv.i
    np.testing.assert_allclose((energy & freezing).to_array(), 0.0, atol=1e-15)


def test_growth_runs_and_draws():
    topology, vertices = scenarios.dish(4)
    parcels, phase = scenarios.seeded(vertices)
    states = list(scenarios.frozen(topology, vertices, parcels, phase, scenarios.anisotropy, 2, 1))
    next(render.animate(topology, vertices, states))
