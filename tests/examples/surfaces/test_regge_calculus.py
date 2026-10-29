"""Regge edge metric basis, rotor holonomy defects, Gauss–Bonnet, and intrinsic Laplace–Beltrami flow."""

import numpy as np

from numga.sparse import spdiag
from examples.surfaces.regge_calculus import core, scenarios


def test_regge_metric_basis_side_readings_heron_and_corner_rotors():
    corners, _ = scenarios.single_triangle(8)
    sides = corners[[2, 0, 1]] - corners[[1, 2, 0]]
    basis, gradients, patch = core.regge_metric_basis(corners)

    assert basis.gatype == core.Metric
    assert gradients.gatype == core.Plane
    assert patch.gatype == core.Patch
    np.testing.assert_allclose(gradients.sum(axis=0).kernel, 0.0, atol=1e-14)

    # Each Regge basis map pairs to 1 on its own edge and to 0 on the other two edges.
    side_readings = basis[:, None](sides[None, :]) & sides[None, :]
    np.testing.assert_allclose(side_readings.kernel, np.eye(3)[..., None], atol=1e-14)

    # With Euclidean squared edge lengths, the Regge metric recovers the ambient duality and reference area.
    euclidean_metric, euclidean_area, euclidean_rotors, euclidean_turns = core.triangle_geometry(
        corners, sides.norm_squared(),
    )
    np.testing.assert_allclose((euclidean_metric - core.Point.dual_inverse()).kernel, 0.0, atol=1e-14)
    np.testing.assert_allclose((euclidean_area - patch.norm()).kernel, 0.0, atol=1e-14)
    np.testing.assert_allclose((euclidean_turns.sum(axis=0) - np.pi * core.mv.xy).kernel, 0.0, atol=1e-14)
    np.testing.assert_allclose(
        (euclidean_rotors[0] * euclidean_rotors[1] * euclidean_rotors[2]).kernel,
        [-1.0, 0.0], atol=1e-14,
    )

    # With stretched squared edge lengths, the wedge pairing matches Heron's formula and corner rotors multiply to -1.
    squared_lengths = sides.norm_squared() * scenarios.edge_stretches
    metric, intrinsic_area, corner_rotors, corner_turns = core.triangle_geometry(corners, squared_lengths)
    assert metric.gatype == core.Metric
    assert intrinsic_area.gatype == core.Scalar
    assert corner_rotors.gatype == core.Rotor
    assert corner_turns.gatype == core.Bivector

    np.testing.assert_allclose(((metric(sides) & sides) - squared_lengths).kernel, 0.0, atol=1e-14)
    lengths = np.sqrt(squared_lengths.kernel[..., 0])
    semiperimeter = lengths.sum() / 2
    heron_area = np.sqrt(semiperimeter * np.prod(semiperimeter - lengths))
    np.testing.assert_allclose(intrinsic_area.kernel, heron_area, atol=1e-14)
    np.testing.assert_allclose(corner_rotors.scalar_norm_squared().kernel, 1.0, atol=1e-14)
    np.testing.assert_allclose((corner_turns.sum(axis=0) - np.pi * core.mv.xy).kernel, 0.0, atol=1e-14)
    np.testing.assert_allclose(
        (corner_rotors[0] * corner_rotors[1] * corner_rotors[2]).kernel,
        [-1.0, 0.0], atol=1e-14,
    )


def test_gauss_bonnet_curvature_defects_and_cotangent_inverse_metric():
    mesh, vertices = scenarios.disk(1.0, 6)
    edges = mesh.d[0] * vertices
    midpoints = mesh.incidence[1, 0].mean(axis=0) * vertices
    edge_squared_lengths = (
        edges.norm_squared()
        + scenarios.radial_rates * midpoints.dot(edges).squared()
        + scenarios.hoop_rates * (midpoints & edges).norm_squared()
    )

    metrics, intrinsic_areas, defects, curvatures, conductivities = core.mesh_geometry(
        mesh, vertices, edge_squared_lengths,
    )
    assert metrics.batch().gatype == core.Metric
    assert intrinsic_areas.batch().gatype == core.Scalar
    assert defects.batch().gatype == core.Bivector
    assert curvatures.batch().gatype == core.Bivector
    assert conductivities.batch().gatype == core.Conductivity

    # Discrete Gauss-Bonnet holds to machine precision across flat, positively curved, and negatively curved metrics.
    np.testing.assert_allclose((defects.sites.sum() - 2 * np.pi * core.mv.xy).kernel, 0.0, atol=1e-12)

    np.testing.assert_allclose(defects.batch()[0, mesh.interior(0)].kernel, 0.0, atol=1e-13)
    np.testing.assert_allclose(curvatures.batch()[0, mesh.interior(0)].kernel, 0.0, atol=1e-11)

    # Scaling squared radial and circumferential distances by one plus `radial_rates` and `hoop_rates` times the squared
    # radius gives the continuum surface a Gaussian curvature of `radial_rates - 3 * hoop_rates` at its centre.
    centre_curvatures = (curvatures.batch()[:, 0] & core.mv.xy.dual()).to_array()
    expected_centre = scenarios.radial_rates - 3.0 * scenarios.hoop_rates
    np.testing.assert_allclose(centre_curvatures, expected_centre, rtol=4.5e-2, atol=1e-12)

    # The side-dyad cotangent formula equals the area-weighted inverse metric on every triangle and every case.
    inverse_conductivities = intrinsic_areas * metrics.batch().inverse().field()
    np.testing.assert_allclose((conductivities - inverse_conductivities).kernel, 0.0, atol=1e-13)

    # Conformal invariance in 2D: scaling all squared edge lengths by a constant factor scales areas by that
    # factor while leaving the area-weighted inverse metric and vertex defects invariant.
    conformal_scale = 2.7
    _, scaled_areas, scaled_defects, _, scaled_conductivities = core.mesh_geometry(
        mesh, vertices, conformal_scale * edge_squared_lengths,
    )
    np.testing.assert_allclose((scaled_areas - conformal_scale * intrinsic_areas).kernel, 0.0, atol=1e-13)
    np.testing.assert_allclose((scaled_defects - defects).kernel, 0.0, atol=1e-13)
    np.testing.assert_allclose((scaled_conductivities - conductivities).kernel, 0.0, atol=1e-13)


def test_intrinsic_laplace_beltrami_cotangent_equivalence_and_flow():
    mesh, vertices = scenarios.disk(1.0, 6)
    d0 = mesh.d[0] * core.Scalar
    edges = mesh.d[0] * vertices
    midpoints = mesh.incidence[1, 0].mean(axis=0) * vertices
    edge_squared_lengths = (
        edges.norm_squared()
        + scenarios.radial_rates * midpoints.dot(edges).squared()
        + scenarios.hoop_rates * (midpoints & edges).norm_squared()
    )
    metrics, intrinsic_areas, _, _, conductivities = core.mesh_geometry(mesh, vertices, edge_squared_lengths)

    sides = mesh.boundary[2, 1] * edges
    patches = (sides[0] & sides[1]) / 2
    corner_gradients = sides / (2 * patches)
    gradient = (spdiag(corner_gradients) * mesh.incidence[2, 0]).sum(axis=0) * core.Scalar
    conductance = gradient.adjugate()(spdiag(conductivities)(gradient))
    np.testing.assert_allclose((conductance - conductance.adjugate()).cells.kernel, 0.0, atol=1e-13)

    # Pulling the Regge inverse metric back through the Whitney face gradient matches the intrinsic cotangent Laplacian.
    dots = -(metrics[:, None](sides[[1, 2, 0]]) & sides[[2, 0, 1]])
    cotangents = dots / (2 * intrinsic_areas[:, None])
    unit_faces = core.as_scalar(np.ones(len(mesh.faces))).field().dual()
    edge_cotangents = (~mesh.incidence[2, 1] * ((cotangents / 2) * unit_faces)).sum(axis=-1)
    cotangent_laplacian = -d0.adjugate()(spdiag(edge_cotangents * core.Scalar)(d0))
    quadratic = (core.mv.x & vertices).squared() - 0.35 * (core.mv.y & vertices).squared()
    np.testing.assert_allclose(
        (conductance(quadratic) - cotangent_laplacian(quadratic)).kernel,
        0.0, atol=1e-13,
    )

    # Steady electrode conduction balances intrinsic flux at every free vertex and bends flow by curvature.
    boundary = mesh.exterior(0)
    abscissa = (core.mv.x & vertices).batch()
    electrodes = boundary[abscissa[boundary].norm() > scenarios.electrode_threshold]
    pinned = mesh.reading(electrodes)
    prescribed = ~pinned * (pinned * (core.mv.x & vertices))
    potential, current, streamfunction = core.intrinsic_conduction(
        mesh, vertices, conductivities, electrodes, prescribed,
    )

    free_vertices = np.setdiff1d(np.arange(mesh.n_vertices), electrodes)
    free = ~mesh.reading(free_vertices) * core.Scalar
    nodal_source = conductance(potential)
    np.testing.assert_allclose(free.adjugate()(nodal_source).kernel, 0.0, atol=1e-11)
    np.testing.assert_allclose(nodal_source.sites.sum().kernel, 0.0, atol=1e-11)

    # Shortening circumferential distances reduces centre current relative to lengthening them.
    centres = mesh.incidence[2, 0].mean(axis=0) * vertices
    origin_face = centres.norm().batch().argmin()
    centre_speeds = current.batch()[:, origin_face].norm().to_array()
    assert centre_speeds[1] < centre_speeds[0] < centre_speeds[2]
    assert np.isfinite(streamfunction.kernel).all()
