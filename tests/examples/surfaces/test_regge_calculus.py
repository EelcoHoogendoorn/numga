"""Coverage tests check Regge edge-metric duality maps, rotor holonomy defects, intrinsic conduction, and figure rendering."""

import matplotlib.pyplot as plt
import numpy as np

from numga import stack
from numga.sparse import spdiag
from examples.surfaces.regge_calculus import core, render, scenarios
from examples.util import shapes


def test_regge_metric_basis_heron_and_corner_rotors():
    corners, sample_points = scenarios.single_triangle(4)
    basis = core.metric_basis(core.triangle, corners)
    squared_lengths = (
        (core.triangle.d[0] * corners).norm_squared()
        * core.mv.scalar(scenarios.edge_stretches[:, None]).field()
    )
    metric = (spdiag(basis) * core.triangle.incidence[2, 1]).sum(axis=0) * squared_lengths
    area = core.intrinsic_areas(core.triangle, corners, metric)
    rotors = core.corner_rotors(core.triangle, corners, metric, area)
    turns = rotors.log()

    assert basis.batch().gatype == core.Metric
    assert metric.batch().gatype == core.Metric
    assert area.batch().gatype == core.Scalar
    assert rotors.batch().gatype == core.Rotor
    assert turns.batch().gatype == core.Bivector

    sides = (core.triangle.boundary[2, 1] * (core.triangle.d[0] * corners)).batch()[:, 0]
    side_readings = basis.batch()[:, 0, None](sides[None, :]) & sides[None, :]
    assert ((side_readings - np.eye(3)).abs() < 1e-10).all()

    prescribed_side_lengths = (core.triangle.incidence[2, 1] * squared_lengths).batch()[:, 0]
    assert (((metric.batch()[0](sides) & sides) - prescribed_side_lengths).abs() < 1e-10).all()

    side_lengths = prescribed_side_lengths.square_root()
    semiperimeter = side_lengths.sum() / 2
    heron_area = (
        semiperimeter
        * (semiperimeter - side_lengths[0])
        * (semiperimeter - side_lengths[1])
        * (semiperimeter - side_lengths[2])
    ).square_root()
    assert ((area.batch()[0] - heron_area).abs() < 1e-10).all()
    assert ((rotors.batch()[:, 0].scalar_norm_squared() - 1).abs() < 1e-10).all()
    assert ((turns.batch()[:, 0].sum(axis=0) - np.pi * core.mv.xy).norm() < 1e-10).all()
    assert ((rotors.batch()[0, 0] * rotors.batch()[1, 0] * rotors.batch()[2, 0] + 1).norm() < 1e-10).all()

    figure = render.metric_basis_figure(corners, sample_points, basis, metric)
    assert isinstance(figure, plt.Figure)
    plt.close(figure)


def test_gauss_bonnet_mirror_covariance_and_conduction():
    topology, coordinates = shapes.hexagonal_disk(1.0, 2)
    V, E, F = topology.n_elements
    vertices = core.mv.vector(coordinates).dual().field()
    edges = topology.d[0] * vertices
    midpoints = topology.incidence[1, 0].mean(axis=0) * vertices
    edge_squared_lengths = (
        edges.norm_squared()
        + scenarios.radial_rates * (midpoints | edges).squared()
        + scenarios.hoop_rates * (midpoints & edges).norm_squared()
    )

    basis = core.metric_basis(topology, vertices)
    metrics = (spdiag(basis) * topology.incidence[2, 1]).sum(axis=0) * edge_squared_lengths
    areas = core.intrinsic_areas(topology, vertices, metrics)
    turns = core.corner_rotors(topology, vertices, metrics, areas).log()
    defects = core.vertex_defects(topology, vertices, turns)
    dual_areas = topology.incidence[2, 0].mean(axis=0).adjugate() * areas
    curvatures = defects / dual_areas
    conductivities = core.triangle_conductivities(topology, vertices, metrics, areas)

    assert ((defects.sites.sum() - 2 * np.pi * core.mv.xy).norm() < 1e-10).all()
    assert ((topology.reading(topology.interior[0], 0) * defects[0]).batch().norm() < 1e-10).all()
    assert (curvatures.batch()[1, 0] & core.mv.xy.dual()) > 0
    assert (curvatures.batch()[2, 0] & core.mv.xy.dual()) < 0

    # Mirroring the reference triangulation mirrors the vertex angle defects and face currents.
    mirror = core.mv.x
    mirrored_vertices = mirror >> vertices
    mirrored_basis = core.metric_basis(topology, mirrored_vertices)
    mirrored_metrics = (spdiag(mirrored_basis) * topology.incidence[2, 1]).sum(axis=0) * edge_squared_lengths
    mirrored_areas = core.intrinsic_areas(topology, mirrored_vertices, mirrored_metrics)
    mirrored_turns = core.corner_rotors(topology, mirrored_vertices, mirrored_metrics, mirrored_areas).log()
    mirrored_defects = core.vertex_defects(topology, mirrored_vertices, mirrored_turns)
    assert (((mirror >> defects.batch()) - mirrored_defects.batch()).norm() < 1e-10).all()

    basis_planes = stack((core.mv.x, core.mv.y))
    identity_residual = (
        metrics.batch()[..., None](conductivities.batch()[..., None](basis_planes))
        - areas.batch()[..., None] * basis_planes
    )
    assert (identity_residual.norm() < 1e-10).all()

    rim_indices = topology.exterior[0]
    rim = topology.reading(rim_indices, 0)
    electrodes = rim_indices[(core.mv.x & (rim * vertices)).abs() > 0.6]
    electrode_reading = topology.reading(electrodes, 0)
    boundary_potential = electrode_reading.adjugate() * (electrode_reading * (core.mv.x & vertices))
    free_indices = np.setdiff1d(np.arange(V), electrodes)

    gradient = core.simplex_gradient(topology, vertices)
    conductance = gradient.adjugate()(conductivities(gradient))
    potential = core.steady_potential(topology, conductance, free_indices, boundary_potential)
    reference_sides = topology.boundary[2, 1] * edges
    reference_areas = ((reference_sides[0] & reference_sides[1]) / 2).antinorm()
    current = -conductivities(gradient(potential)) / reference_areas

    assert ((topology.reading(free_indices, 0) * conductance(potential)).batch().norm() < 1e-9).all()
    assert (conductance(potential).sites.sum().norm() < 1e-9).all()

    centres = topology.incidence[2, 0].mean(axis=0) * vertices
    curvature_figure = render.curvature_defects_figure(topology, vertices, centres, metrics, curvatures)
    flow_figure = render.intrinsic_flow_figure(topology, vertices, potential, current)
    assert isinstance(curvature_figure, plt.Figure)
    assert isinstance(flow_figure, plt.Figure)
    plt.close(curvature_figure)
    plt.close(flow_figure)
