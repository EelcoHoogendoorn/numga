"""Coverage tests for reproducing kernels, skinning modes, hyperelastic stress, and surface rendering."""

from itertools import product

import matplotlib.pyplot as plt
import numpy as np

from numga import stack
from numga.topology import Topology
from examples.mechanics.freeform import core, geometry, render, weights
from examples.mechanics.freeform.numerics import minimized_state


def cube_samples():
    centers = core.mv.vector(list(product((-0.7, 0.7), repeat=3))).dual()
    queries = core.mv.vector(list(product((-0.45, 0, 0.45), repeat=3))).dual()
    volumes = np.full(len(queries), 1 / len(queries))
    corners = core.mv.vector([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]).dual()
    return queries, centers, volumes, corners


def test_reproducing_kernels_and_material_weighted_skinning_modes():
    mv = core.mv
    queries, centers, volumes, _ = cube_samples()
    n_queries = len(queries)
    n_modes = 4
    radius = 1.1
    evaluation, gradient = weights.kernels(queries, centers, radius)
    slope = 0.3 * mv.x - 0.7 * mv.y + 0.2 * mv.z
    nodal = (1.2 + (slope & centers)).field()
    assert ((evaluation(nodal) - (1.2 + (slope & queries))).abs() < 1e-10).all()
    assert ((gradient(nodal) - slope.broadcast_to(n_queries)).scalar_norm_squared() < 1e-18).all()

    # A quadratic field tests the spatial derivative of the reproducing correction beyond affine patches.
    quadratic_nodal = ((mv.x & centers) * (mv.y & centers)).field()
    heading = mv.vector([0.3, -0.7, 0.2]).dual()
    step = 1e-5
    ahead, _ = weights.kernels(queries + step * heading, centers, radius)
    behind, _ = weights.kernels(queries - step * heading, centers, radius)
    finite_difference = (ahead(quadratic_nodal) - behind(quadratic_nodal)) / (2 * step)
    assert ((finite_difference - (gradient(quadratic_nodal) & heading)).abs() < 1e-7).all()

    lame_lambda = np.linspace(1, 2, n_queries)
    lame_mu = np.linspace(0.5, 1.5, n_queries)
    eigenvalues, modes = weights.skinning_modes(evaluation, gradient, volumes, lame_lambda, lame_mu, n_modes)
    mode_values = evaluation[:, None](modes)
    mode_gradients = gradient[:, None](modes)
    mass_gram = (mode_values[:, :, None] * mode_values[:, None, :] * volumes[:, None, None]).sum(axis=0)
    stiffness_gram = (
        mode_gradients[:, :, None].scalar_product(mode_gradients[:, None, :])
        * (volumes * (lame_lambda + 4 * lame_mu))[:, None, None]
    ).sum(axis=0)
    identity = mv.scalar(np.eye(n_modes)[..., None])
    diagonal_eigenvalues = mv.scalar(np.diag(eigenvalues.to_array())[..., None])
    assert ((mass_gram - identity).abs() < 1e-10).all()
    assert ((stiffness_gram - diagonal_eigenvalues).abs() < 1e-9).all()
    assert (mode_gradients[:, 0].scalar_norm_squared() < 1e-18).all()


def test_hyperelastic_stress_work_and_frame_indifference():
    mv, Antivector = core.mv, core.Antivector
    x_dir, y_dir, z_dir = core.directions
    deformed = (Antivector + 0.3 * x_dir * (mv.y & Antivector)
                - 0.1 * y_dir * (mv.z & Antivector) + 0.2 * z_dir * (mv.z & Antivector))
    flat = Antivector - z_dir * (mv.z & Antivector)
    local = stack((deformed, flat))
    lame_lambda, lame_mu = 3.2, 1.7
    energy, stress = core.constitutive(local, lame_lambda, lame_mu)
    perturbation = (0.2 * z_dir * (mv.x & Antivector)
                    - 0.4 * y_dir * (mv.y & Antivector) + 0.3 * z_dir * (mv.z & Antivector))
    step = 1e-5
    ahead, _ = core.constitutive(local + step * perturbation, lame_lambda, lame_mu)
    behind, _ = core.constitutive(local - step * perturbation, lame_lambda, lame_mu)
    finite_difference = (ahead - behind) / (2 * step)
    work = (stress[:, None](core.planes) & perturbation(core.directions)).sum(axis=-1)
    assert ((finite_difference - work).abs() < 1e-7).all()

    # A rigid rotation turns the stress output frame without changing elastic energy.
    rotor = (0.3 * mv.xy + 0.2 * mv.yz).exp()
    placed_local = rotor >> local
    placed_energy, placed_stress = core.constitutive(placed_local, lame_lambda, lame_mu)
    turned_stress = rotor >> stress
    assert ((placed_energy - energy).abs() < 1e-10).all()
    assert ((placed_stress - turned_stress).scalar_norm_squared().contract() < 1e-18).all()


def test_reduced_forces_metric_whitening_and_surface_render():
    mv = core.mv
    queries, centers, volumes, corners = cube_samples()
    n_queries = len(queries)
    radius = 1.1
    constant_mode = 1 + (mv.x & centers) * 0
    curved_mode = (mv.x & centers) * (mv.y & centers)
    modes = stack((constant_mode, curved_mode)).field()
    sampling, gradients = core.deformation_basis(queries, centers, radius, modes, corners)
    rng = np.random.default_rng(4)
    n_controls = len(corners) * len(modes)
    state = mv.vector(rng.normal(scale=0.04, size=(n_controls, 3))).dual().field()
    perturbation = mv.vector(rng.normal(scale=0.1, size=(n_controls, 3))).dual().field()
    clamp = mv.scalar(np.linspace(0.1, 0.5, n_queries)[:, None])
    load = mv.vector(rng.normal(scale=0.03, size=(n_queries, 3)))
    lame_lambda, lame_mu = 3.2, 1.7

    def evaluate_potential(trial_state):
        return core.potential(trial_state, sampling, gradients, volumes, lame_lambda, lame_mu, clamp, load)

    _, force = evaluate_potential(state)
    step = 1e-5
    ahead, _ = evaluate_potential(state + step * perturbation)
    behind, _ = evaluate_potential(state - step * perturbation)
    work = (force & perturbation).sites.sum()
    assert (((ahead - behind) / (2 * step) - work).abs() < 1e-7).all()

    metric = core.control_mass(sampling, volumes)

    def quadratic_objective(trial_state):
        error = trial_state - state
        restoring_force = metric(error)
        return (restoring_force & error).sites.sum() / 2, restoring_force

    solution = minimized_state(state * 0, quadratic_objective, metric, 20, 1e-10)
    assert ((solution - state).scalar_norm_squared().sites.sum() < 1e-16).all()

    # Exercise volume sampling and GPU surface rendering on a single closed tetrahedron.
    tet_topology = Topology(4, [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]])
    tet_vertices = corners
    samples, sample_volumes = geometry.volume_samples(tet_topology, tet_vertices.field(), 8)
    assert len(samples) == 8 and sample_volumes.shape == (8,)
    figure = render.surface_figure(tet_vertices, tet_topology)
    plt.close(figure)
    frame = next(render.surface_frames(tet_topology, [tet_vertices]))
    assert frame.ndim == 3 and frame.shape[-1] == 3
