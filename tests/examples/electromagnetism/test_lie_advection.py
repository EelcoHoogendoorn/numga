"""Coverage tests check Cartan world-sheet decomposition, gauge invariance, conservation laws, and rendering for Lie advection."""

import matplotlib.pyplot as plt
import numpy as np

from numga import stack
from examples.electromagnetism.lie_advection import core, render, scenarios
from examples.util import shapes


def test_cartan_decomposition_and_ribbon_figure():
    endpoint_differences, swept_fluxes, world_sheets, face_field = core.cartan_decomposition(
        scenarios.triangle_corners, scenarios.corner_potentials, scenarios.corner_velocities,
    )

    assert endpoint_differences.batch().gatype == core.Scalar
    assert swept_fluxes.batch().gatype == core.Scalar
    assert world_sheets.batch().gatype == core.Patch
    assert face_field.batch().gatype == core.Field

    # Regressive associativity equates world-sheet flux to the interior-product line integral,
    # and endpoint readings cancel pairwise around the closed triangle boundary.
    triangle_sides = core.triangle.boundary[2, 1] * (core.triangle.d[0] * scenarios.triangle_corners)
    edge_motions = core.triangle.incidence[2, 1] * (
        core.triangle.incidence[1, 0].mean(axis=0) * (core.time + scenarios.corner_velocities)
    )
    interior_integrals = (face_field & edge_motions) & triangle_sides
    assert ((swept_fluxes - interior_integrals).abs() < 1e-12).all()
    assert (endpoint_differences.sum(axis=0).abs() < 1e-12).all()

    figure = render.cartan_ribbon_figure(
        scenarios.triangle_corners, scenarios.corner_potentials, scenarios.corner_velocities,
        scenarios.ribbon_step, endpoint_differences, swept_fluxes,
    )
    assert len(figure.axes) == 2
    plt.close(figure)


def test_uniform_field_stationarity_and_gauge_invariance():
    topology, coordinates = shapes.hexagonal_disk(1.0, 3)
    vertices = core.mv(core.Upright, coordinates).dual().field()
    curl = core.whitney_curl(topology, vertices)
    stream = core.vortex_streamfunction(vertices, scenarios.swirl_rate, scenarios.ellipticity)
    swept_patches = core.swept_edge_patches(topology, vertices, stream)
    advection = core.advection_map(topology, vertices, curl, swept_patches)
    curl_stiffness = core.curl_stiffness_map(topology, vertices, curl)
    curl_laplacian = core.curl_laplacian_map(topology, curl_stiffness)

    d0 = topology.d[0] * core.Scalar
    d1 = topology.d[1] * core.Scalar
    assert (((core.mv.xy & swept_patches) + d0(stream)).abs() < 1e-12).all()
    assert (d1(core.mv.xy & swept_patches).abs() < 1e-12).all()

    # A uniform magnetic 2-form produces an exact gradient edge rate, keeping face flux stationary.
    edges = topology.d[0] * vertices
    edge_midpoints = topology.incidence[1, 0].mean(axis=0) * vertices
    uniform_strength = 1.35
    uniform_cochain = (uniform_strength / 2) * ((core.mv.xy & edge_midpoints) & edges)
    assert ((curl(uniform_cochain) - uniform_strength * core.mv.xy).norm() < 1e-12).all()
    uniform_rate = advection(uniform_cochain)
    assert ((uniform_rate + uniform_strength * d0(stream)).abs() < 1e-12).all()
    assert (d1(uniform_rate).abs() < 1e-12).all()

    # Adding an exact 0-form gradient leaves curl, stiffness, Laplacian, and evolution rate invariant.
    cochain = core.initial_cochain(
        topology, vertices, scenarios.lobe_centres, scenarios.lobe_strengths, scenarios.lobe_radius,
    )
    gauge = 0.8 * (core.mv.x & vertices).squared() - 0.55 * (core.mv.y & vertices)
    gauged_cochain = cochain + d0(gauge)
    assert ((d1(gauged_cochain) - d1(cochain)).abs() < 1e-12).all()
    assert ((curl(gauged_cochain) - curl(cochain)).norm() < 1e-12).all()
    assert ((curl_stiffness(gauged_cochain) - curl_stiffness(cochain)).antinorm() < 1e-11).all()
    assert ((curl_laplacian(gauged_cochain) - curl_laplacian(cochain)).abs() < 1e-11).all()
    rate = advection - curl_laplacian * scenarios.resistivities[:, None]
    assert ((rate(gauged_cochain) - rate(cochain)).abs() < 1e-11).all()


def test_conservation_dissipation_and_advection_renders():
    topology, coordinates = shapes.hexagonal_disk(1.0, 3)
    V, E, F = topology.n_elements
    vertices = core.mv(core.Upright, coordinates).dual().field()
    curl = core.whitney_curl(topology, vertices)
    stream = core.vortex_streamfunction(vertices, scenarios.swirl_rate, scenarios.ellipticity)
    swept_patches = core.swept_edge_patches(topology, vertices, stream)
    advection = core.advection_map(topology, vertices, curl, swept_patches)
    curl_stiffness = core.curl_stiffness_map(topology, vertices, curl)
    curl_laplacian = core.curl_laplacian_map(topology, curl_stiffness)
    flux_density = core.flux_density_map(topology, vertices)

    rng = np.random.default_rng(7)
    random_cochain = core.mv.scalar(rng.standard_normal((E, 1))).field()
    random_energy = (random_cochain & curl_stiffness(random_cochain)).sites.sum()
    ideal_rate = advection(random_cochain)
    assert (((ideal_rate & curl_stiffness(random_cochain)).sites.sum() / random_energy).abs() < 1e-12).all()
    resistive_rate = -curl_laplacian(random_cochain)
    assert (resistive_rate & curl_stiffness(random_cochain)).sites.sum() < 0

    start_cochain = core.initial_cochain(
        topology, vertices, scenarios.lobe_centres, scenarios.lobe_strengths, scenarios.lobe_radius,
    )
    batched_start = stack((start_cochain, start_cochain))
    rate = advection - curl_laplacian * scenarios.resistivities[:, None]
    n_test_steps = 4
    cochain_states = stack(list(core.cochain_trajectory(batched_start, rate, scenarios.dt, n_test_steps)))
    d1 = topology.d[1] * core.Scalar
    energies = (cochain_states & curl_stiffness(cochain_states)).sites.sum() / 2
    total_fluxes = d1(cochain_states).sites.sum()

    assert ((total_fluxes - total_fluxes[0]).abs() < 1e-12).all()
    assert (((energies[:, 0] / energies[0, 0]) - 1).abs() < 1e-5).all()
    assert ((energies[1:, 1] - energies[:-1, 1]) < 0).all()

    times = np.arange(n_test_steps + 1) * scenarios.dt
    comparison_fig = render.advection_comparison_figure(
        topology, vertices, flux_density(cochain_states[-1]), times, energies, total_fluxes,
    )
    assert len(comparison_fig.subfigs) == 2
    plt.close(comparison_fig)

    rendered_frames = list(render.advection_frames(topology, vertices, flux_density, cochain_states[:2]))
    assert len(rendered_frames) == 2
    assert rendered_frames[0].ndim == 3
