"""Coverage tests for reproducing kernels, skinning modes, hyperelastic stress, and surface rendering."""

from itertools import product

import matplotlib.pyplot as plt
import numpy as np

from numga import dim, stack
from numga.topology import Topology
from examples.mechanics.freeform import core, geometry, render, weights
from examples.mechanics.freeform.numerics import minimized_state

force_unit = dim.mass * dim.length / dim.time**2
energy_unit = force_unit * dim.length
stress_unit = force_unit / dim.length**2
stiffness_unit = force_unit / dim.length**4
load_unit = force_unit / dim.length**3


def cube_samples():
    centers = core.mv.vector(list(product((-0.7, 0.7), repeat=3)), unit=dim.length).dual()
    queries = core.mv.vector(list(product((-0.45, 0, 0.45), repeat=3)), unit=dim.length).dual()
    volumes = core.mv.scalar(np.full((len(queries), 1), 1 / len(queries)), unit=dim.length**3)
    corners = core.mv.vector([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], unit=dim.length).dual()
    return queries, centers, volumes, corners


def test_reproducing_kernels_and_material_weighted_skinning_modes():
    mv = core.mv
    queries, centers, volumes, _ = cube_samples()
    n_queries = len(queries)
    n_modes = 4
    radius = mv.scalar(1.1, unit=dim.length)
    evaluation, gradient = weights.kernels(queries, centers, radius)
    slope = (0.3 * mv.x - 0.7 * mv.y + 0.2 * mv.z) / dim.length
    nodal = (1.2 + (slope & centers)).field()
    assert ((evaluation(nodal) - (1.2 + (slope & queries))).abs() < 1e-10).all()
    assert ((gradient(nodal) - slope.broadcast_to(n_queries)).scalar_norm_squared() < mv.scalar(1e-18, unit=dim.length**-2)).all()

    # A quadratic field tests the spatial derivative of the reproducing correction beyond affine patches.
    quadratic_nodal = ((mv.x & centers) * (mv.y & centers) / dim.length**2).field()
    heading = mv.vector([0.3, -0.7, 0.2]).dual()
    step = mv.scalar(1e-5, unit=dim.length)
    ahead, _ = weights.kernels(queries + step * heading, centers, radius)
    behind, _ = weights.kernels(queries - step * heading, centers, radius)
    finite_difference = (ahead(quadratic_nodal) - behind(quadratic_nodal)) / (2 * step)
    assert ((finite_difference - (gradient(quadratic_nodal) & heading)).abs() < mv.scalar(1e-7, unit=dim.length**-1)).all()

    lame_lambda = mv.scalar(np.linspace(1, 2, n_queries)[:, None], unit=stress_unit)
    lame_mu = mv.scalar(np.linspace(0.5, 1.5, n_queries)[:, None], unit=stress_unit)
    eigenvalues, modes = weights.skinning_modes(evaluation, gradient, volumes, lame_lambda, lame_mu, n_modes)
    assert eigenvalues.unit == stiffness_unit
    assert modes.unit is dim.none
    mode_values = evaluation[:, None](modes)
    mode_gradients = gradient[:, None](modes)
    volume_weights = volumes / volumes.sum(axis=0)
    mass_gram = (mode_values[:, :, None] * mode_values[:, None, :] * volume_weights[:, None, None]).sum(axis=0)
    stiffness_gram = (
        mode_gradients[:, :, None].scalar_product(mode_gradients[:, None, :])
        * (volume_weights * (lame_lambda + 4 * lame_mu))[:, None, None]
    ).sum(axis=0)
    identity = mv.scalar(np.eye(n_modes)[..., None])
    diagonal_eigenvalues = mv.scalar(np.diag((eigenvalues / stiffness_unit).to_array())[..., None], unit=stiffness_unit)
    assert ((mass_gram - identity).abs() < 1e-10).all()
    assert ((stiffness_gram - diagonal_eigenvalues).abs() < mv.scalar(1e-9, unit=stiffness_unit)).all()
    assert (mode_gradients[:, 0].scalar_norm_squared() < mv.scalar(1e-18, unit=dim.length**-2)).all()


def test_hyperelastic_stress_work_and_frame_indifference():
    mv, Antivector = core.mv, core.Antivector
    x_dir, y_dir, z_dir = core.directions
    deformed = (Antivector + 0.3 * x_dir * (mv.y & Antivector)
                - 0.1 * y_dir * (mv.z & Antivector) + 0.2 * z_dir * (mv.z & Antivector))
    flat = Antivector - z_dir * (mv.z & Antivector)
    local = stack((deformed, flat))
    lame_lambda = mv.scalar(3.2, unit=stress_unit)
    lame_mu = mv.scalar(1.7, unit=stress_unit)
    energy, stress = core.constitutive(local, lame_lambda, lame_mu)
    assert energy.unit == stress_unit
    assert stress.unit == stress_unit
    perturbation = (0.2 * z_dir * (mv.x & Antivector)
                    - 0.4 * y_dir * (mv.y & Antivector) + 0.3 * z_dir * (mv.z & Antivector))
    step = 1e-5
    ahead, _ = core.constitutive(local + step * perturbation, lame_lambda, lame_mu)
    behind, _ = core.constitutive(local - step * perturbation, lame_lambda, lame_mu)
    finite_difference = (ahead - behind) / (2 * step)
    work = (stress[:, None](core.planes) & perturbation(core.directions)).sum(axis=-1)
    assert ((finite_difference - work).abs() < mv.scalar(1e-7, unit=stress_unit)).all()

    # A rigid rotation turns the stress output frame without changing elastic energy.
    rotor = (0.3 * mv.xy + 0.2 * mv.yz).exp()
    placed_local = rotor >> local
    placed_energy, placed_stress = core.constitutive(placed_local, lame_lambda, lame_mu)
    turned_stress = rotor >> stress
    assert ((placed_energy - energy).abs() < mv.scalar(1e-10, unit=stress_unit)).all()
    assert ((placed_stress - turned_stress).scalar_norm_squared().contract() < mv.scalar(1e-18, unit=stress_unit**2)).all()


def test_reduced_forces_metric_whitening_and_surface_render():
    mv = core.mv
    queries, centers, volumes, corners = cube_samples()
    n_queries = len(queries)
    radius = mv.scalar(1.1, unit=dim.length)
    constant_mode = 1 + (mv.x & centers) / dim.length * 0
    curved_mode = (mv.x & centers) * (mv.y & centers) / dim.length**2
    modes = stack((constant_mode, curved_mode)).field()
    sampling, gradients = core.deformation_basis(queries, centers, radius, modes, corners)
    rng = np.random.default_rng(4)
    n_controls = len(corners) * len(modes)
    state = mv.vector(rng.normal(scale=0.04, size=(n_controls, 3)), unit=dim.length).dual().field()
    perturbation = mv.vector(rng.normal(scale=0.1, size=(n_controls, 3)), unit=dim.length).dual().field()
    clamp = mv.scalar(np.linspace(0.1, 0.5, n_queries)[:, None], unit=stiffness_unit)
    load = mv.vector(rng.normal(scale=0.03, size=(n_queries, 3)), unit=load_unit)
    lame_lambda = mv.scalar(3.2, unit=stress_unit)
    lame_mu = mv.scalar(1.7, unit=stress_unit)

    def evaluate_potential(trial_state):
        return core.potential(trial_state, sampling, gradients, volumes, lame_lambda, lame_mu, clamp, load)

    energy, force = evaluate_potential(state)
    assert energy.unit == energy_unit
    assert force.unit == force_unit
    step = 1e-5
    ahead, _ = evaluate_potential(state + step * perturbation)
    behind, _ = evaluate_potential(state - step * perturbation)
    work = (force & perturbation).sites.sum()
    assert (((ahead - behind) / (2 * step) - work).abs() < mv.scalar(1e-7, unit=energy_unit)).all()

    density = mv.scalar(1.0, unit=dim.mass / dim.length**3)
    time_step = mv.scalar(1.0, unit=dim.time)
    metric = core.control_mass(sampling, volumes * density) / time_step**2

    def quadratic_objective(trial_state):
        error = trial_state - state
        restoring_force = metric(error)
        return (restoring_force & error).sites.sum() / 2, restoring_force

    solution = minimized_state(state * 0, quadratic_objective, metric, 20, 1e-10)
    assert solution.unit == dim.length
    assert ((solution - state).scalar_norm_squared().sites.sum() < mv.scalar(1e-16, unit=dim.length**2)).all()

    # Exercise volume sampling and GPU surface rendering on a single closed tetrahedron.
    tet_topology = Topology(4, [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]])
    tet_vertices = corners
    samples, sample_volumes = geometry.volume_samples(tet_topology, tet_vertices.field(), 8)
    assert len(samples) == 8 and sample_volumes.shape == (8,) and sample_volumes.unit == dim.length**3
    figure = render.surface_figure(tet_vertices, tet_topology)
    plt.close(figure)
    frame = next(render.surface_frames(tet_topology, [tet_vertices]))
    assert frame.ndim == 3 and frame.shape[-1] == 3
