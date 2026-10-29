"""Barycentric hats, edge Whitney 1-forms, exact midpoint quadrature, and anisotropic conduction."""

import numpy as np

from numga import stack
from numga.sparse import spdiag
from examples.surfaces.whitney_forms import core, scenarios


def test_barycentric_hats_and_edge_whitney_forms_on_single_triangle_and_diamond():
    corners, points = scenarios.single_triangle(8)
    hats, gradients = core.barycentric_hats(corners, points)
    corner_hats, _ = core.barycentric_hats(corners, corners)

    assert hats.gatype == core.Scalar
    assert gradients.gatype == core.Plane
    np.testing.assert_allclose(corner_hats.kernel, np.eye(3)[..., None], atol=1e-14)
    np.testing.assert_allclose(hats.sum(axis=0).kernel, 1.0, atol=1e-14)
    np.testing.assert_allclose(gradients.sum(axis=0).kernel, 0.0, atol=1e-14)

    # Barycentric interpolation reproduces any affine function and its gradient exactly.
    true_gradient = 1.4 * core.mv.x - 0.85 * core.mv.y
    corner_values = 0.6 + (true_gradient & corners)
    sample_values = 0.6 + (true_gradient & points)
    interpolated = (corner_values[:, None] * hats).sum(axis=0)
    reconstructed_gradient = (corner_values * gradients).sum(axis=0)
    np.testing.assert_allclose(interpolated.kernel, sample_values.kernel, atol=1e-14)
    np.testing.assert_allclose(reconstructed_gradient.kernel, true_gradient.kernel, atol=1e-14)

    # Edge 1-forms match Whitney's alternating formula and satisfy Kronecker duality along the edges.
    edge_forms = core.edge_whitney_forms(corners, points)
    first, second = [1, 2, 0], [2, 0, 1]
    alternating_forms = hats[first] * gradients[second, None] - hats[second] * gradients[first, None]
    np.testing.assert_allclose((edge_forms - alternating_forms).kernel, 0.0, atol=1e-14)

    sides = corners[[2, 0, 1]] - corners[[1, 2, 0]]
    for fraction in (0.15, 0.5, 0.82):
        edge_points = corners[first] * (1 - fraction) + corners[second] * fraction
        readings = core.edge_whitney_forms(corners, edge_points) & sides[None, :]
        np.testing.assert_allclose(readings.kernel, np.eye(3)[..., None], atol=1e-14)

    # Across the shared edge of a two-triangle diamond, tangential readings match while normal readings jump.
    mesh, vertices, _ = scenarios.edge_diamond(6)
    diamond = core.Operators(mesh, vertices)
    tails, heads = mesh.incidence[1, 0] * vertices
    shared_tail, = tails.batch()[mesh.interior_edges]
    shared_head, = heads.batch()[mesh.interior_edges]
    shared_edge = shared_head - shared_tail
    fractions = np.linspace(0.1, 0.9, 7)
    seam_points = shared_tail * (1 - fractions) + shared_head * fractions
    # The forms of the shared edge on each face, signed by the sense in which the face runs along it.
    seam_forms = (seam_points[:, None] - diamond.corners[None, :]) / (2 * diamond.patches)
    seam_whitney1 = (spdiag(seam_forms) * mesh.boundary[2, 1]).sum(axis=1) * core.Scalar
    shared_circulation = core.as_scalar(np.isin(np.arange(len(mesh.edges)), mesh.interior_edges)).field()
    seams = seam_whitney1(shared_circulation)
    np.testing.assert_allclose((seams & shared_edge).kernel, 1.0, atol=1e-14)
    left_seam, right_seam = seams.batch()[:, 0], seams.batch()[:, 1]
    normal_jump = (left_seam - right_seam) & core.mv.x.dual()
    assert np.max(np.abs(normal_jump.kernel)) > 0.5


def test_cochain_complex_and_exact_midpoint_quadrature_including_obtuse_triangles():
    mesh, vertices = scenarios.disk(1.0, 5)
    operators = core.Operators(mesh, vertices)

    assert operators.d0.gatype == core.ga.gatype((core.Scalar, core.Scalar))
    assert operators.d1.gatype == core.ga.gatype((core.Scalar, core.Scalar))
    assert operators.gradient.gatype == core.ga.gatype((core.Plane, core.Scalar))
    assert operators.whitney1.gatype == core.ga.gatype((core.Plane, core.Scalar))
    assert operators.curl.gatype == core.ga.gatype((core.Bivector, core.Scalar))
    assert operators.whitney2.gatype == core.ga.gatype((core.Bivector, core.Scalar))
    assert not np.any(operators.d1(operators.d0).cells.kernel)

    # Whitney reconstruction commutes with the exterior derivative on 0-cochains and 1-cochains.
    potential = (core.mv.x & vertices) * 0.9 - (core.mv.y & vertices) * 1.3
    np.testing.assert_allclose(
        (operators.whitney1(operators.d0(potential)) - operators.gradient(potential)).kernel,
        0.0, atol=1e-14,
    )
    cochain = operators.d0(potential) + core.as_scalar(np.cos(np.arange(len(mesh.edges)))).field()
    np.testing.assert_allclose(
        (operators.curl(cochain) - operators.whitney2(operators.d1(cochain))).kernel,
        0.0, atol=1e-12,
    )
    face_cochain = operators.d1(cochain)
    np.testing.assert_allclose(
        (operators.patches & operators.whitney2(face_cochain)).kernel,
        face_cochain.kernel, atol=1e-13,
    )

    # Three-midpoint quadrature matches interior degree-2 Gauss quadrature and pulls back to the 0-form conductance.
    fibre = (core.mv.xy * (-scenarios.fibre_angle / 2)).exp() >> core.mv.x.dual()
    quarter_turn = (core.mv.xy * (-np.pi / 4)).exp()
    directions = stack((fibre, quarter_turn >> fibre))
    conductivity = (directions * (core.Plane & directions) * scenarios.principal_conductivities).sum(axis=-1)
    edge_conductances = core.edge_conductance(operators, conductivity)

    barycentric_interior = np.array([[2 / 3, 1 / 6, 1 / 6], [1 / 6, 2 / 3, 1 / 6], [1 / 6, 1 / 6, 2 / 3]])
    interior_points = (operators.corners[None, :] * barycentric_interior).sum(axis=1)
    interior_forms = (interior_points[:, None] - operators.corners[None, :]) / (2 * operators.patches)
    interior_whitney1 = (spdiag(interior_forms) * mesh.boundary[2, 1]).sum(axis=1) * core.Scalar
    weights = spdiag(((operators.areas / 3) * conductivity)[..., None])
    interior_conductances = interior_whitney1.adjugate()(weights(interior_whitney1)).sum(axis=-1)
    np.testing.assert_allclose((edge_conductances - interior_conductances).cells.kernel, 0.0, atol=1e-14)
    np.testing.assert_allclose((edge_conductances - edge_conductances.adjugate()).cells.kernel, 0.0, atol=1e-14)

    conductance = operators.gradient.adjugate()(spdiag(operators.areas * conductivity)(operators.gradient))
    quadratic = (core.mv.x & vertices).squared() - 0.4 * (core.mv.y & vertices).squared()
    np.testing.assert_allclose(
        (operators.d0.adjugate()(edge_conductances(operators.d0(quadratic))) - conductance(quadratic)).kernel,
        0.0, atol=1e-13,
    )

    # On the obtuse diamond, circumcentric DEC's shared-edge weight is negative while Whitney's 1-form mass is positive-definite.
    obtuse_mesh, obtuse_vertices = scenarios.obtuse_diamond()
    obtuse_ops = core.Operators(obtuse_mesh, obtuse_vertices)
    next_sides, prev_sides = obtuse_ops.sides[[1, 2, 0]], obtuse_ops.sides[[2, 0, 1]]
    cotangents = -next_sides.dot(prev_sides) / (2 * obtuse_ops.areas)
    unit_amount = core.as_scalar(np.ones(len(obtuse_mesh.faces))).field().dual()
    cotangent_weights = (~obtuse_mesh.incidence[2, 1] * ((cotangents / 2) * unit_amount)).sum(axis=0)
    assert (cotangent_weights.batch()[0] & core.mv.scalar([1.0])).to_array() < -1.0

    obtuse_edge_conductance = core.edge_conductance(obtuse_ops, core.Plane.dual())
    for edge in range(len(obtuse_mesh.edges)):
        unit_edge = core.as_scalar(np.arange(len(obtuse_mesh.edges)) == edge).field()
        assert -(obtuse_edge_conductance(unit_edge) & unit_edge).sites.sum() > 0


def test_isotropic_and_anisotropic_conduction_patch_test_and_flux_balance():
    mesh, vertices = scenarios.disk(1.0, 6)
    operators = core.Operators(mesh, vertices)
    fibre = (core.mv.xy * (-scenarios.fibre_angle / 2)).exp() >> core.mv.x.dual()
    quarter_turn = (core.mv.xy * (-np.pi / 4)).exp()
    directions = stack((fibre, quarter_turn >> fibre))
    conductivity = (directions * (core.Plane & directions) * scenarios.principal_conductivities).sum(axis=-1)

    # Linear patch test: prescribing a linear potential on the entire boundary recovers it everywhere for any conductivity.
    boundary = mesh.exterior(0)
    applied_field = 1.25 * core.mv.x - 0.6 * core.mv.y
    exact_potential = applied_field & vertices
    rim = mesh.reading(boundary)
    prescribed_rim = ~rim * (rim * exact_potential)
    potential, current, _ = core.conduction(operators, conductivity, boundary, prescribed_rim)
    np.testing.assert_allclose(potential.batch()[0].kernel, exact_potential.kernel, atol=1e-13)
    np.testing.assert_allclose(potential.batch()[1].kernel, exact_potential.kernel, atol=1e-13)
    expected_current = -conductivity(applied_field)
    np.testing.assert_allclose(current.batch()[0].kernel, np.broadcast_to(expected_current[0].kernel, (len(mesh.faces), 2)), atol=1e-12)
    np.testing.assert_allclose(current.batch()[1].kernel, np.broadcast_to(expected_current[1].kernel, (len(mesh.faces), 2)), atol=1e-12)

    # Electrode solve balances flux at every free vertex and conserves total current between the two electrodes.
    abscissa = (core.mv.x & vertices).batch()
    electrodes = boundary[abscissa[boundary].norm() > scenarios.electrode_threshold]
    pinned = mesh.reading(electrodes)
    prescribed = ~pinned * (pinned * (core.mv.x & vertices))
    potential, current, streamfunction = core.conduction(operators, conductivity, electrodes, prescribed)

    conductance = operators.gradient.adjugate()(spdiag(operators.areas * conductivity)(operators.gradient))
    free_vertices = np.setdiff1d(np.arange(mesh.n_vertices), electrodes)
    free = ~mesh.reading(free_vertices) * core.Scalar
    nodal_source = conductance(potential)
    np.testing.assert_allclose(free.adjugate()(nodal_source).kernel, 0.0, atol=1e-11)
    np.testing.assert_allclose(nodal_source.sites.sum().kernel, 0.0, atol=1e-11)
    assert np.isfinite(streamfunction.kernel).all()
