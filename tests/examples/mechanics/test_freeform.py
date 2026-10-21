"""Affine reproduction, skinning modes and finite-deformation elastic work."""

from itertools import product

import numpy as np

from numga import stack
from examples.mechanics.freeform import core, weights
from examples.mechanics.freeform.numerics import minimize_state


def cube_samples():
    centers = core.mv.vector(list(product((-0.7, 0.7), repeat=3))).dual()
    queries = core.mv.vector(list(product((-0.45, 0, 0.45), repeat=3))).dual()
    volumes = np.full(len(queries), 1 / len(queries))
    corners = core.mv.vector([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]).dual()
    return queries, centers, volumes, corners


def test_reproducing_kernels_and_material_weighted_skinning_modes():
    mv = core.mv
    queries, centers, volumes, _ = cube_samples()
    radius = 1.1
    evaluation, gradient = weights.kernels(queries, centers, radius)
    slope = 0.3 * mv.x - 0.7 * mv.y + 0.2 * mv.z
    nodal = (1.2 + (slope & centers)).field()
    np.testing.assert_allclose(evaluation(nodal).kernel, (1.2 + (slope & queries)).kernel, atol=1e-10)
    np.testing.assert_allclose(gradient(nodal).kernel, slope.broadcast_to(len(queries)).kernel, atol=1e-10)

    # A non-affine field tests the changing kernel correction, beyond the reproduced patch.
    nodal = ((mv.x & centers) * (mv.y & centers)).field()
    heading = mv.vector([0.3, -0.7, 0.2]).dual()
    step = 1e-5
    ahead, _ = weights.kernels(queries + step * heading, centers, radius)
    behind, _ = weights.kernels(queries - step * heading, centers, radius)
    finite = (ahead(nodal) - behind(nodal)) / (2 * step)
    np.testing.assert_allclose(finite.kernel, (gradient(nodal) & heading).kernel, atol=1e-7, rtol=1e-7)

    lame_lambda = np.linspace(1, 2, len(queries))
    lame_mu = np.linspace(0.5, 1.5, len(queries))
    eigenvalues, modes = weights.modes(evaluation, gradient, volumes, lame_lambda, lame_mu, 4)
    samples = evaluation[:, None](modes)
    changes = gradient[:, None](modes)
    mass = (samples[:, :, None] * samples[:, None, :] * volumes[:, None, None]).sum(axis=0)
    stiffness = (changes[:, :, None].scalar_product(changes[:, None, :])
                 * (volumes * (lame_lambda + 4 * lame_mu))[:, None, None]).sum(axis=0)
    np.testing.assert_allclose(mass.kernel[..., 0], np.eye(4), atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(stiffness.kernel[..., 0], np.diag(eigenvalues.kernel[..., 0]),
                               atol=1e-9, rtol=1e-9)
    np.testing.assert_allclose(changes[:, 0].kernel, 0, atol=1e-10)


def test_hyperelastic_stress_is_energy_derivative_even_at_flat_collapse():
    mv, Antivector = core.mv, core.Antivector
    x, y, z = core.directions
    deformed = Antivector + 0.3 * x * (mv.y & Antivector) - 0.1 * y * (mv.z & Antivector)
    deformed = deformed + 0.2 * z * (mv.z & Antivector)
    flat = Antivector - z * (mv.z & Antivector)
    local = stack((deformed, flat))
    lame_lambda, lame_mu = 3.2, 1.7
    energy, stress = core.constitutive(local, lame_lambda, lame_mu)
    change = 0.2 * z * (mv.x & Antivector) - 0.4 * y * (mv.y & Antivector)
    change = change + 0.3 * z * (mv.z & Antivector)
    step = 1e-5
    ahead, _ = core.constitutive(local + step * change, lame_lambda, lame_mu)
    behind, _ = core.constitutive(local - step * change, lame_lambda, lame_mu)
    finite = (ahead - behind) / (2 * step)
    # The work of the stress: the force on each plane across the axes, paired with the change along its axis.
    work = (stress[:, None](core.planes) & change(core.directions)).sum(axis=-1)
    np.testing.assert_allclose(finite.kernel, work.kernel, atol=1e-7, rtol=1e-7)

    # A rigid placement changes the stress's output frame without changing elastic energy.
    rotor = (0.3 * mv.xy + 0.2 * mv.yz).exp()
    placed_energy, placed_stress = core.constitutive(rotor >> local, lame_lambda, lame_mu)
    np.testing.assert_allclose(placed_energy.kernel, energy.kernel, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(placed_stress.kernel, (rotor >> stress).kernel, atol=1e-10, rtol=1e-10)


def test_reduced_forces_match_energy_and_metric_whitening_preserves_the_solution():
    mv = core.mv
    queries, centers, volumes, corners = cube_samples()
    radius = 1.1
    constant = 1 + (mv.x & centers) * 0
    curved = (mv.x & centers) * (mv.y & centers)
    modes = stack((constant, curved)).field()
    sampling, gradients = core.deformation_basis(queries, centers, radius, modes, corners)
    rng = np.random.default_rng(4)
    n_controls = len(corners) * len(modes)
    state = mv.vector(rng.normal(scale=0.04, size=(n_controls, 3))).dual().field()
    change = mv.vector(rng.normal(scale=0.1, size=(n_controls, 3))).dual().field()
    clamp = mv.scalar(np.linspace(0.1, 0.5, len(queries))[:, None])
    load = mv.vector(rng.normal(scale=0.03, size=(len(queries), 3)))
    lame_lambda, lame_mu = 3.2, 1.7

    def potential(value):
        return core.potential(value, sampling, gradients, volumes, lame_lambda, lame_mu, clamp, load)

    _, force = potential(state)
    step = 1e-5
    ahead, _ = potential(state + step * change)
    behind, _ = potential(state - step * change)
    work = (force & change).sites.sum()
    np.testing.assert_allclose(((ahead - behind) / (2 * step)).kernel, work.kernel, atol=1e-7, rtol=1e-7)

    metric = core.mass(sampling, volumes)

    # The optimizer must preserve the physical field solution through its coordinate change.
    def objective(value):
        error = value - state
        force = metric(error)
        return (force & error).sites.sum() / 2, force

    result = minimize_state(state * 0, objective, metric, 20, 1e-10)
    np.testing.assert_allclose(result.kernel, state.kernel, atol=1e-9, rtol=1e-9)
