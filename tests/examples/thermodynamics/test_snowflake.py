"""A seed of ice in undercooled water grows six arms along the crystal's axes from its six-fold anisotropy, and none
without it, on a mesh with no six-fold symmetry of its own."""
import jax
import numpy as np

from examples.thermodynamics.snowflake import core, scenarios
from examples.util import shapes

mv = core.mv
# A small dish, with the scenario's spacing of its mesh, and the steps the seed grows for, until its arms show.
dish_radius, n_divisions, n_steps = 1.5, 50, 400


def small_dish():
    """The small dish's topology, its vertices' places across and up `[vertices, 2]`, and its vertices."""
    topology, coordinates = shapes.concentric_disk(np.array([dish_radius]), n_divisions)
    vertices = (core.volume + core.across * coordinates[:, 0] + core.up * coordinates[:, 1]).field()
    return topology, coordinates, vertices


def six_fold_moment(anisotropy: float) -> float:
    """The mean over the vertices in ice of the cosine of six times their angle from the crystal's axis."""
    topology, coordinates, vertices = small_dish()
    _, grown = list(scenarios.frozen(topology, vertices, *scenarios.seeded(coordinates), anisotropy, 2, n_steps))[-1]
    frozen = np.asarray(grown > 0.5)
    angles = np.arctan2(coordinates[:, 1], coordinates[:, 0]) - np.pi / 2
    return np.cos(6 * angles[frozen]).mean()


def test_six_arms_come_from_the_anisotropy():
    assert six_fold_moment(scenarios.anisotropy) > 0.3
    assert abs(six_fold_moment(0.0)) < 0.1


def test_freezing_keeps_the_energy():
    """Freezing moves energy from the latent store into the heat: the energy plane pairs with it to nothing, and over a
    run the heat gained is the latent heat of the ice gained, before any heat reaches the dish's rim."""
    freezing = core.ice + scenarios.latent_heat * core.heat
    energy = mv.h - scenarios.latent_heat * mv.i
    np.testing.assert_allclose((energy & freezing).to_array(), 0.0, atol=1e-15)


def test_drive_is_the_entropys_slope():
    """The phase's drive, the push to freeze written as a pairing with freezing and the edge's slope written out with
    the rotor's fifth power, is what `jax.grad` finds for the total entropy at fixed energy, times the absolute melting
    temperature, on a seed mid-growth."""
    topology, coordinates, vertices = small_dish()
    parcels, phase = list(scenarios.frozen(topology, vertices, *scenarios.seeded(coordinates), scenarios.anisotropy,
                                           2, 100))[-1]
    areas, gradients, cells = core.triangles(topology, vertices)
    gradient, spreading, _ = core.operators(topology, areas, gradients)
    freezing = core.ice + scenarios.latent_heat * core.heat
    edge = (scenarios.crystal_axis, scenarios.mean_width, scenarios.anisotropy, scenarios.regularization,
            scenarios.well_height)
    thermal = (scenarios.latent_heat, scenarios.melting_temperature, scenarios.absolute_zero)

    def entropy(trial):
        # The parcels at fixed energy holding the ice this phase holds, its latent heat with it.
        trial_parcels = parcels + (core.held_ice(trial) - core.held_ice(phase)) * freezing
        return core.total_entropy(areas, cells, gradient, spreading, *edge, *thermal, trial_parcels, trial).to_array()

    absolute_melting = scenarios.melting_temperature - scenarios.absolute_zero
    found = absolute_melting * jax.grad(entropy)(phase).to_array()
    written = core.phase_drive(topology, areas, gradients, cells, gradient, spreading, *edge, *thermal, freezing,
                               parcels, phase).to_array()
    np.testing.assert_allclose(written, found, atol=1e-12 * np.abs(found).max())
