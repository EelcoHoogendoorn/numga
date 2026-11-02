"""A seed of ice in undercooled water grows six arms along the crystal's axes from its six-fold anisotropy, and none
without it, on a mesh with no six-fold symmetry of its own."""
import jax
import numpy as np

from examples.thermodynamics.snowflake import core, scenarios
from examples.util import shapes

mv = core.mv
# A small dish, with the scenario's spacing of its mesh, and the steps the seed grows for, until its arms show.
dish_radius, n_divisions, n_steps = 1.5, 50, 400


def six_fold_moment(anisotropy: float) -> float:
    """The mean over the vertices in ice of the cosine of six times their angle from the crystal's axis."""
    topology, coordinates = shapes.concentric_disk(np.array([dish_radius]), n_divisions)
    vertices = (core.volume + core.across * coordinates[:, 0] + core.up * coordinates[:, 1]).field()
    grown = list(scenarios.frozen(topology, vertices, scenarios.seeded(coordinates), anisotropy, 2, n_steps))[-1]
    frozen = np.asarray((mv.i & grown) > 0.5)
    angles = np.arctan2(coordinates[:, 1], coordinates[:, 0]) - np.pi / 2
    return np.cos(6 * angles[frozen]).mean()


def test_six_arms_come_from_the_anisotropy():
    assert six_fold_moment(scenarios.anisotropy) > 0.2
    assert abs(six_fold_moment(0.0)) < 0.1


def test_freezing_keeps_the_energy():
    """Freezing moves energy from the latent store into the heat: the energy plane pairs with it to nothing, and over a
    run the heat gained is the latent heat of the ice gained, before any heat reaches the dish's rim."""
    freezing = core.ice + scenarios.latent_heat * core.heat
    energy = mv.h - scenarios.latent_heat * mv.i
    np.testing.assert_allclose((energy & freezing).to_array(), 0.0, atol=1e-15)


def test_written_out_slope_is_the_free_energys_derivative():
    """The free energy's derivative at each vertex, written out with the rotor's fifth power, is what `jax.grad` finds
    for the free energy itself, on a seed mid-growth."""
    topology, coordinates = shapes.concentric_disk(np.array([dish_radius]), n_divisions)
    vertices = (core.volume + core.across * coordinates[:, 0] + core.up * coordinates[:, 1]).field()
    grown = list(scenarios.frozen(topology, vertices, scenarios.seeded(coordinates), scenarios.anisotropy, 2, 100))[-1]
    fraction, temperatures = mv.i & grown, mv.h & grown
    areas, gradients, cells = core.triangles(topology, vertices)
    gradient, spreading, _ = core.operators(topology, areas, gradients)
    parameters = (scenarios.crystal_axis, scenarios.mean_width, scenarios.anisotropy, scenarios.regularization,
                  scenarios.tilt_strength, scenarios.tilt_steepness, scenarios.melting_temperature)

    def free_energy(trial):
        steepness = gradient(trial)
        drive = -(steepness ^ mv.i)
        squared = drive & spreading(drive)
        third_power = (steepness * scenarios.crystal_axis) ** 3
        widths = scenarios.mean_width * (1 + scenarios.anisotropy * third_power.scalar_product(third_power)
                                         / (squared + scenarios.regularization ** 2) ** 3)
        tilt = scenarios.tilt_strength / np.pi * (scenarios.tilt_steepness
                                                  * (scenarios.melting_temperature - temperatures)).arctan()
        well = trial ** 2 * (1 - trial) ** 2 / 4 - tilt * trial ** 2 * (3 - 2 * trial) / 6
        return ((areas * widths ** 2 * squared / 2).sites.sum() + (cells * well).sites.sum()).to_array()

    found = jax.grad(free_energy)(fraction).to_array()
    written = core.free_energy_slope(topology, areas, gradients, cells, gradient, spreading, *parameters, fraction,
                                     temperatures).to_array()
    np.testing.assert_allclose(written, found, atol=1e-12 * np.abs(found).max())
