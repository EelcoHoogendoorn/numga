"""Typed discrete derivatives, viscous channel flow and independence of edge directions."""

import numpy as np

from numga.sparse import SparseExtensor
from examples.fluids.dec_stokes import core, scenarios


def flow(mesh, vertices, inlet, outlet):
    operators = core.Operators(mesh, vertices)
    ordinate = core.mv.y & vertices
    streamfunction = 1.5 * (ordinate - 4 * ordinate.squared() * ordinate / (3 * scenarios.HEIGHT**2))
    ends = SparseExtensor.selection(core.mv.context, np.concatenate((inlet, outlet)), len(mesh.edges))
    prescribed = ~ends * (ends * (mesh.d0 * streamfunction))
    boundary_circulation = core.mv.scalar(np.zeros((mesh.n_vertices, 1))).field()
    result = core.stokes(operators, scenarios.VISCOSITY, prescribed, mesh.boundary_edges, boundary_circulation)
    return operators, prescribed, result


def test_incidence_is_topological_and_hodges_supply_geometry_and_duality():
    mesh, vertices, *_ = scenarios.channel(8)
    operators = core.Operators(mesh, vertices)
    for derivative in (operators.d0, operators.d1):
        assert derivative.gatype == core.ga.gatype((core.Scalar, core.Scalar))
        np.testing.assert_array_equal(np.unique(derivative.cells.kernel), [-1, 1])
    assert not np.any(operators.d1(operators.d0).cells.kernel)

    # Scaling and stretching geometry leave every oriented boundary sum unchanged.
    scale = 1.7
    scaled = core.Operators(mesh, scale * vertices)
    stretch = core.mv.x.dual() * (1.2 * core.mv.x & core.Point) + core.mv.y.dual() * (0.85 * core.mv.y & core.Point)
    stretched = core.Operators(mesh, stretch(vertices))
    for changed in (scaled, stretched):
        np.testing.assert_array_equal(changed.d0.cells.kernel, operators.d0.cells.kernel)
        np.testing.assert_array_equal(changed.d1.cells.kernel, operators.d1.cells.kernel)
    np.testing.assert_allclose(scaled.star0.cells.kernel, scale**2 * operators.star0.cells.kernel, atol=1e-11)
    np.testing.assert_allclose(scaled.star1.cells.kernel, operators.star1.cells.kernel, atol=1e-11)
    np.testing.assert_allclose(scaled.star2.cells.kernel, operators.star2.cells.kernel / scale**2, atol=1e-11)
    np.testing.assert_allclose(stretched.areas.kernel, 1.2 * 0.85 * operators.areas.kernel, atol=1e-11)
    assert np.max(np.abs(stretched.h1.kernel - operators.h1.kernel)) > 0.01

    # Circumcentres are equidistant from each triangle's corners; dual weights are length ratios.
    corners = vertices.batch()[mesh.faces.T].field()
    radii_squared = (corners - operators.circumcentres).norm_squared()
    np.testing.assert_allclose((radii_squared - radii_squared[0]).kernel, 0, atol=1e-11)
    ratio = operators.dual_edges.norm() / operators.edges.norm()
    np.testing.assert_allclose(operators.h1.kernel, ratio.kernel, atol=1e-11)
    assert np.all(operators.h1.kernel > 0)

    # Constant flow reads positive circulation along the displayed dual direction.
    potential = (core.mv.y & vertices) - 0.3 * (core.mv.x & vertices)
    circulation = ((core.mv.x + 0.3 * core.mv.y) & operators.dual_edges).dual()
    np.testing.assert_allclose(operators.star1(operators.d0(potential)).kernel,
                               circulation.kernel, atol=1e-11)

    for hodge in (operators.star0, operators.star1, operators.star2):
        assert hodge.gatype == core.ga.gatype((core.Amount, core.Scalar))
    vertex_values = core.mv.scalar(np.cos(np.arange(mesh.n_vertices))[:, None]).field()
    edge_values = core.mv.scalar(np.sin(np.arange(len(mesh.edges)))[:, None]).field()
    dual_edges = operators.star1(edge_values)
    np.testing.assert_allclose(operators.inverse_star0(operators.star0(vertex_values)).kernel,
                               vertex_values.kernel, atol=1e-11)
    np.testing.assert_allclose(operators.star1.solve(dual_edges).kernel, edge_values.kernel, atol=1e-11)
    np.testing.assert_allclose((edge_values & dual_edges).kernel,
                               (edge_values.squared() * ratio).kernel, atol=1e-11)

    # The incidence's adjugate reads the same primal-dual work from either side.
    edge_work = (operators.d0(vertex_values) & dual_edges).sites.sum()
    vertex_work = (vertex_values & operators.d0.adjugate()(dual_edges)).sites.sum()
    np.testing.assert_allclose(edge_work.kernel, vertex_work.kernel, atol=1e-11)


def test_straight_channel_converges_to_poiseuille_velocity_pressure_and_vorticity():
    errors = []
    for n_divisions in (8, 16):
        mesh, vertices, inlet, outlet, _ = scenarios.plain_channel(n_divisions)
        operators, _, (_, pressure, vorticity, velocity) = flow(mesh, vertices, inlet, outlet)
        ordinate = core.mv.y & operators.face_centres
        exact_velocity = core.mv.x.dual() * 1.5 * (1 - 4 * ordinate.squared() / scenarios.HEIGHT**2)
        exact_pressure = -12 * scenarios.VISCOSITY * (core.mv.x & operators.circumcentres) / scenarios.HEIGHT**2
        exact_pressure -= (exact_pressure * operators.areas).sites.sum() / operators.areas.sites.sum()
        exact_vorticity = core.mv.xy * (12 * (core.mv.y & vertices) / scenarios.HEIGHT**2)

        velocity_error = (((velocity - exact_velocity).norm_squared() * operators.areas).sites.sum()
                          / operators.areas.sites.sum()).square_root()
        pressure_error = (((pressure - exact_pressure).squared() * operators.areas).sites.sum()
                          / operators.areas.sites.sum()).square_root()
        vorticity_error = (((vorticity - exact_vorticity).norm_squared() * operators.dual_areas).sites.sum()
                           / operators.dual_areas.sites.sum()).square_root()
        errors.append([velocity_error.to_array(), pressure_error.to_array(), vorticity_error.to_array()])

    coarse, fine = np.array(errors)
    assert np.all(fine < 0.75 * coarse)
    assert np.all(fine < [0.05, 0.08, 0.04])


def test_obstacle_flow_balances_every_cell_and_preserves_boundary_flux():
    mesh, vertices, inlet, outlet, walls = scenarios.channel(12)
    operators, prescribed, (flux, pressure, vorticity, velocity) = flow(mesh, vertices, inlet, outlet)
    np.testing.assert_allclose((mesh.d1 * flux).kernel, 0, atol=1e-10)
    np.testing.assert_allclose(flux.batch()[mesh.boundary_edges].kernel,
                               prescribed.batch()[mesh.boundary_edges].kernel, atol=1e-11)
    np.testing.assert_allclose(flux.batch()[walls].kernel, 0, atol=1e-11)

    # Summing oriented face boundaries cancels every interior edge and reads the net throughflow.
    one = core.mv.scalar(np.ones((len(mesh.faces), 1))).field()
    boundary = ~mesh.d1 * one
    outgoing = (boundary * flux).batch().to_array()
    np.testing.assert_allclose(outgoing[inlet].sum(), -scenarios.HEIGHT, atol=1e-11)
    np.testing.assert_allclose(outgoing[outlet].sum(), scenarios.HEIGHT, atol=1e-11)

    abscissa = (core.mv.x & operators.face_centres).to_array()
    pressure_values = pressure.to_array()
    assert pressure_values[abscissa < 0.5].mean() > pressure_values[abscissa > scenarios.LENGTH - 0.5].mean()
    assert (vorticity.norm_squared() * operators.dual_areas).sites.sum().to_array() > 0
    assert np.isfinite(velocity.kernel).all()


def test_reversing_mesh_edges_changes_readings_and_preserves_physical_flow():
    mesh, vertices, inlet, outlet, _ = scenarios.channel(12)
    _, _, result = flow(mesh, vertices, inlet, outlet)
    reversed_mesh = core.Mesh(mesh.n_vertices, mesh.faces.copy())
    reversed_edges = np.arange(len(mesh.edges)) % 3 == 0
    reversed_mesh.edges[reversed_edges] = reversed_mesh.edges[reversed_edges, ::-1]
    reversed_mesh.face_edge_orientation *= np.where(reversed_edges[reversed_mesh.face_edges], -1, 1)
    _, _, changed = flow(reversed_mesh, vertices, inlet, outlet)

    signs = core.mv.scalar(np.where(reversed_edges, -1, 1)[:, None]).field()
    np.testing.assert_allclose(changed[0].kernel, (result[0] * signs).kernel, atol=1e-10)
    for before, after in zip(result[1:], changed[1:]):
        np.testing.assert_allclose(after.kernel, before.kernel, atol=1e-9)
