"""A seed of ice in undercooled water: freezing keeps the energy, and the seed grows for a step on a coarse dish, and
its first frame drawn."""
import numpy as np

from examples.thermodynamics.snowflake import core, render, scenarios

mv = core.mv


def test_freezing_keeps_the_energy():
    """Freezing moves energy from the latent store into the heat: the energy plane pairs with it to nothing, and over a
    run the heat gained is the latent heat of the ice gained, before any heat reaches the dish's rim."""
    freezing = core.ice + scenarios.latent_heat * core.heat
    energy = mv.h - scenarios.latent_heat * mv.i
    np.testing.assert_allclose((energy & freezing).to_array(), 0.0, atol=1e-15)


def test_growth_runs_and_draws():
    topology, coordinates, vertices = scenarios.dish(n_divisions=4)
    parcels, phase = scenarios.seeded(coordinates)
    states = list(scenarios.frozen(topology, vertices, parcels, phase, scenarios.anisotropy, 2, 1))
    next(render.animate(topology, vertices, states))
