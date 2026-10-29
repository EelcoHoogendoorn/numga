"""Edge world sheets, Cartan's formula, and structure-preserving Lie advection of magnetic flux in spacetime."""

import numpy as np

from numga import stack
from numga.sparse import spdiag
from examples.electromagnetism.lie_advection import core, scenarios


def test_cartan_magic_formula_and_edge_world_sheets_on_moving_triangle():
    corners, _ = scenarios.single_triangle(6)
    endpoint_differences, swept_fluxes, world_sheets, face_bivector = core.cartan_decomposition(
        corners, scenarios.corner_forms, scenarios.corner_velocities,
    )

    assert endpoint_differences.gatype == core.Scalar
    assert swept_fluxes.gatype == core.Scalar
    assert world_sheets.gatype == core.Patch
    assert face_bivector.gatype == core.Bivector

    # Associativity of the regressive pairing equates dA & (u & e) to ((dA & u) & e).
    sides = corners[[2, 0, 1]] - corners[[1, 2, 0]]
    corner_motions = core.time + scenarios.corner_velocities
    edge_motions = (corner_motions[[1, 2, 0]] + corner_motions[[2, 0, 1]]) / 2
    interior_integrals = (face_bivector & edge_motions) & sides
    np.testing.assert_allclose((swept_fluxes - interior_integrals).kernel, 0.0, atol=1e-13)

    # Around the closed triangle boundary, the endpoint differences cancel pairwise.
    np.testing.assert_allclose(endpoint_differences.sum(axis=0).kernel, 0.0, atol=1e-15)

    # Cartan's sum matches the exact material time derivative of the edge line integrals.
    patch = (sides[0] & sides[1]) / 2
    gradients = (core.time & sides) / (2 * (core.time & patch))
    plus_corners = corners + 0.5 * corner_motions
    minus_corners = corners - 0.5 * corner_motions
    plus_sides = plus_corners[[2, 0, 1]] - plus_corners[[1, 2, 0]]
    minus_sides = minus_corners[[2, 0, 1]] - minus_corners[[1, 2, 0]]
    plus_midpoints = (plus_corners[[1, 2, 0]] + plus_corners[[2, 0, 1]]) / 2
    minus_midpoints = (minus_corners[[1, 2, 0]] + minus_corners[[2, 0, 1]]) / 2
    plus_hats = 1 + (gradients[:, None] & (plus_midpoints[None, :] - corners[:, None]))
    minus_hats = 1 + (gradients[:, None] & (minus_midpoints[None, :] - corners[:, None]))
    plus_forms = (plus_hats * scenarios.corner_forms[:, None]).sum(axis=0)
    minus_forms = (minus_hats * scenarios.corner_forms[:, None]).sum(axis=0)
    material_rates = (plus_forms & plus_sides) - (minus_forms & minus_sides)
    np.testing.assert_allclose(((endpoint_differences + swept_fluxes) - material_rates).kernel, 0.0, atol=1e-13)

    # Total boundary circulation rate equals the divergence of the linear velocity times the face flux.
    plus_patch = (plus_sides[0] & plus_sides[1]) / 2
    minus_patch = (minus_sides[0] & minus_sides[1]) / 2
    face_flux_rate = face_bivector & (plus_patch - minus_patch)
    np.testing.assert_allclose((swept_fluxes.sum(axis=0) - face_flux_rate).kernel, 0.0, atol=1e-13)


def test_world_sheets_uniform_field_preservation_and_gauge_invariance():
    mesh, vertices = scenarios.disk(1.0, 6)
    operators = core.Operators(mesh, vertices)

    assert operators.d0.gatype == core.ga.gatype((core.Scalar, core.Scalar))
    assert operators.d1.gatype == core.ga.gatype((core.Scalar, core.Scalar))
    assert operators.gradient.gatype == core.ga.gatype((core.Plane.derive.within("x y"), core.Scalar))
    assert operators.curl.gatype == core.ga.gatype((core.Magnetic, core.Scalar))
    assert operators.edge_velocity_map.gatype == core.ga.gatype((core.Point, core.Point))
    assert operators.edge_curl.gatype == core.ga.gatype((core.Magnetic, core.Scalar))
    assert operators.curl_stiffness.gatype == core.ga.gatype((core.Amount, core.Scalar))
    assert operators.curl_laplacian.gatype == core.ga.gatype((core.Scalar, core.Scalar))

    stream = core.vortex_streamfunction(vertices, scenarios.swirl_rate, scenarios.ellipticity)
    face_velocity, fixed_sheets, comoving_sheets = core.edge_world_sheets(operators, stream)
    swept_patches = fixed_sheets - comoving_sheets
    np.testing.assert_allclose((operators.gradient(stream) & face_velocity).kernel, 0.0, atol=1e-12)
    np.testing.assert_allclose(((core.mv.xy & swept_patches) + operators.d0(stream)).kernel, 0.0, atol=1e-13)
    np.testing.assert_allclose(operators.d1(core.mv.xy & swept_patches).kernel, 0.0, atol=1e-13)

    # A uniform magnetic 2-form produces an exact gradient edge rate -B0 * d0(stream), so its face flux is stationary.
    uniform_strength = 1.35
    uniform_cochain = (uniform_strength / 2) * ((core.mv.xy & operators.edge_midpoints) & operators.edges)
    np.testing.assert_allclose(
        (operators.curl(uniform_cochain) - uniform_strength * core.mv.xy).kernel,
        0.0, atol=1e-12,
    )
    advection = spdiag(core.Magnetic & swept_patches)(operators.edge_curl)
    uniform_rate = advection(uniform_cochain)
    np.testing.assert_allclose(
        (uniform_rate + uniform_strength * operators.d0(stream)).kernel,
        0.0, atol=1e-12,
    )
    np.testing.assert_allclose(operators.d1(uniform_rate).kernel, 0.0, atol=1e-12)

    # Adding any exact 0-form gradient d0(phi) leaves curl, curl stiffness, curl Laplacian, and the evolution map invariant.
    cochain = core.initial_cochain(operators, scenarios.lobe_centres, scenarios.lobe_strengths, scenarios.lobe_radius)
    gauge = 0.8 * (core.mv.x & vertices).squared() - 0.55 * (core.mv.y & vertices)
    gauged_cochain = cochain + operators.d0(gauge)
    np.testing.assert_allclose((operators.d1(gauged_cochain) - operators.d1(cochain)).kernel, 0.0, atol=1e-14)
    np.testing.assert_allclose((operators.curl(gauged_cochain) - operators.curl(cochain)).kernel, 0.0, atol=1e-12)
    np.testing.assert_allclose(
        (operators.curl_stiffness(gauged_cochain) - operators.curl_stiffness(cochain)).kernel,
        0.0, atol=1e-11,
    )
    np.testing.assert_allclose(
        (operators.curl_laplacian(gauged_cochain) - operators.curl_laplacian(cochain)).kernel,
        0.0, atol=1e-11,
    )
    rate = advection - operators.curl_laplacian * scenarios.resistivities[:, None]
    rate_plain = rate(cochain)
    rate_gauged = rate(gauged_cochain)
    np.testing.assert_allclose((rate_gauged - rate_plain).kernel, 0.0, atol=1e-12)


def test_ideal_flux_and_energy_conservation_versus_resistive_dissipation():
    mesh, vertices = scenarios.disk(1.0, 7)
    operators = core.Operators(mesh, vertices)
    stream = core.vortex_streamfunction(vertices, scenarios.swirl_rate, scenarios.ellipticity)
    _, fixed_sheets, comoving_sheets = core.edge_world_sheets(operators, stream)
    swept_patches = fixed_sheets - comoving_sheets
    start_cochain = core.initial_cochain(
        operators, scenarios.lobe_centres, scenarios.lobe_strengths, scenarios.lobe_radius,
    )

    # For any edge cochain, the ideal advective rate is orthogonal to S1(a), while the resistive rate is dissipative.
    rng = np.random.default_rng(7)
    random_cochain = core.as_scalar(rng.standard_normal(len(mesh.edges))).field()
    random_energy = (random_cochain & operators.curl_stiffness(random_cochain)).sites.sum()
    advection = spdiag(core.Magnetic & swept_patches)(operators.edge_curl)
    ideal_rate = advection(random_cochain)
    np.testing.assert_allclose(
        ((ideal_rate & operators.curl_stiffness(random_cochain)).sites.sum() / random_energy).kernel,
        0.0, atol=1e-12,
    )
    resistive_only = -operators.curl_laplacian(random_cochain)
    assert (resistive_only & operators.curl_stiffness(random_cochain)).sites.sum() < 0

    batched_start = stack((start_cochain, start_cochain))
    rate = advection - operators.curl_laplacian * scenarios.resistivities[:, None]
    states = list(core.advect(batched_start, rate, scenarios.dt, 60))
    energies = stack([(state & operators.curl_stiffness(state)).sites.sum() / 2 for state in states])
    total_fluxes = stack([operators.d1(state).sites.sum() for state in states])

    np.testing.assert_allclose((total_fluxes[:, 0] - total_fluxes[0, 0]).kernel, 0.0, atol=1e-14)
    np.testing.assert_allclose((total_fluxes[:, 1] - total_fluxes[0, 1]).kernel, 0.0, atol=1e-13)
    np.testing.assert_allclose((energies[:, 0] / energies[0, 0]).kernel, 1.0, rtol=1e-5)
    assert np.all(np.diff(energies[:, 1].to_array()) < 0)
