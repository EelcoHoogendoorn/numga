"""Barycentric hats, edge Whitney 1-forms, the forms on a tetrahedron, exact midpoint quadrature, and anisotropic
conduction."""

from functools import reduce
from operator import and_

import numpy as np

from numga import stack
from numga.sparse import spdiag
from examples.surfaces.whitney_forms import core, scenarios
from examples.util import shapes


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
    first, second = core.complements(core.triangle, 0).T
    alternating_forms = hats[first] * gradients[second, None] - hats[second] * gradients[first, None]
    np.testing.assert_allclose((edge_forms - alternating_forms).kernel, 0.0, atol=1e-14)

    sides = (core.triangle.boundary[2, 1] * (core.triangle.d[0] * corners.field())).batch()[:, 0]
    for fraction in (0.15, 0.5, 0.82):
        edge_points = corners[first] * (1 - fraction) + corners[second] * fraction
        readings = core.edge_whitney_forms(corners, edge_points) & sides[None, :]
        np.testing.assert_allclose(readings.kernel, np.eye(3)[..., None], atol=1e-14)

    # Across the shared edge of a two-triangle diamond, tangential readings match while normal readings jump.
    topology, vertices, _ = scenarios.edge_diamond(6)
    diamond = core.Operators(topology, vertices)
    shared = topology.reading(topology.interior[1], 1)                         # Scalar+[1 <- edges]
    shared_tail, shared_head = (shared * (topology.incidence[1, 0] * vertices)).batch()[:, 0]   # [] Point-, Point-
    shared_edge = shared_head - shared_tail
    fractions = np.linspace(0.1, 0.9, 7)
    seam_points = shared_tail * (1 - fractions) + shared_head * fractions
    # The forms of the shared edge on each face, signed by the sense in which the face runs along it.
    seam_forms = (seam_points[:, None] - diamond.corners[None, :]) / (2 * diamond.patches)
    seam_whitney1 = (spdiag(seam_forms) * topology.boundary[2, 1]).sum(axis=1) * core.Scalar
    shared_circulation = ~shared * core.mv.scalar([[1.0]]).field()            # Scalar+[edges]
    seams = seam_whitney1(shared_circulation)
    np.testing.assert_allclose((seams & shared_edge).kernel, 1.0, atol=1e-14)
    left_seam, right_seam = seams.batch()[:, 0], seams.batch()[:, 1]
    normal_jump = (left_seam - right_seam) & core.mv.x.dual()
    assert np.max(np.abs(normal_jump.kernel)) > 0.5


def test_tetrahedral_whitney_0_1_2_forms_and_opposite_edge_helicity():
    topology, vertices, points = scenarios.single_tetrahedron(6)
    forms = [core.tetrahedral_forms(topology, vertices, m) for m in range(3)]
    ramps, edge_forms, face_forms = forms
    hats = ramps[:, None] & points[None, :]

    assert ramps.batch().gatype == core.SpacePlane
    assert hats.batch().gatype == core.SpaceScalar
    assert (edge_forms[:, None] & points[None, :]).batch().gatype == core.SpacePlane
    assert (face_forms[:, None] & points[None, :]).batch().gatype.refines(core.SpaceBivector)
    # Each degree's forms read one on their own simplex's join, in the sense it is stored, and nothing on the others.
    for m, degree_forms in enumerate(forms):
        spans = reduce(and_, topology.on[3, m] * vertices)
        np.testing.assert_allclose((degree_forms[:, None] & spans[None, :]).kernel,
                                   np.eye(spans.shape[0])[..., None, None], atol=1e-13)
    np.testing.assert_allclose(hats.sum(axis=0).kernel, 1.0, atol=1e-13)

    # Barycentric interpolation reproduces any affine potential in space and its gradient exactly.
    true_ramp = 1.3 * core.space_mv.x - 0.75 * core.space_mv.y + 0.55 * core.space_mv.z + 0.4 * core.space_mv.w
    corners = topology.incidence[3, 0] * vertices
    corner_values = true_ramp & corners
    interpolated = (corner_values[:, None] * hats).sum(axis=0)
    np.testing.assert_allclose(interpolated.kernel, (true_ramp & points).kernel, atol=1e-13)
    np.testing.assert_allclose(((corner_values * ramps).sum(axis=0) - true_ramp).kernel, 0.0, atol=1e-13)

    # Whitney's formulas from the ramps: the edge forms, and the face forms, which read one on the join of a face's
    # corners, twice the face.
    tails, heads = topology.corners[3, 1]
    alternating_edge_forms = hats[tails] * ramps[heads, None] - hats[heads] * ramps[tails, None]
    np.testing.assert_allclose(((edge_forms[:, None] & points[None, :]) - alternating_edge_forms).kernel, 0.0, atol=1e-13)
    first, second, third = topology.corners[3, 2]
    alternating_face_forms = 2 * (hats[first] * (ramps[second, None] ^ ramps[third, None])
                                  + hats[second] * (ramps[third, None] ^ ramps[first, None])
                                  + hats[third] * (ramps[first, None] ^ ramps[second, None]))
    np.testing.assert_allclose((2 * (face_forms[:, None] & points[None, :]) - alternating_face_forms).kernel, 0.0, atol=1e-13)
    # Without a point, each form is the wedge of the ramps at its simplex's corners.
    np.testing.assert_allclose((edge_forms - (ramps[tails] ^ ramps[heads])).kernel, 0.0, atol=1e-13)
    np.testing.assert_allclose((face_forms - (ramps[first] ^ ramps[second] ^ ramps[third])).kernel, 0.0, atol=1e-13)

    # The volume pairing of edge forms with curls is the same for a sheared, stretched tetrahedron, and links only
    # opposite edges, with the sense of the tetrahedron's corners in the order the two edges list them, over six.
    tet_helicity = core.tetrahedral_helicity(topology, vertices)
    deformed_helicity = core.tetrahedral_helicity(topology, scenarios.deformed_tetrahedron())
    assert tet_helicity.batch().gatype == core.SpaceScalar
    np.testing.assert_allclose((tet_helicity - deformed_helicity).kernel, 0.0, atol=1e-13)
    edge_corners = topology.corners[3, 1].T
    edge_pairs = np.concatenate(np.broadcast_arrays(edge_corners[:, None], edge_corners[None, :]), axis=-1)
    disjoint = (edge_corners[:, None, :, None] != edge_corners[None, :, None, :]).all(axis=(-1, -2))
    linking = np.where(disjoint, core.simplex_parity(edge_pairs), 0) / 6
    np.testing.assert_allclose(tet_helicity.kernel[..., 0, 0], linking, atol=1e-13)


def test_cochain_complex_and_exact_midpoint_quadrature_including_obtuse_triangles():
    topology, coordinates = shapes.hexagonal_disk(1.0, 5)
    vertices = scenarios.mv.vector(coordinates).dual().field()
    operators = core.Operators(topology, vertices)

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
    cochain = operators.d0(potential) + core.mv.scalar(np.cos(np.arange(topology.n_elements[1]))[..., None]).field()
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
    interior_whitney1 = (spdiag(interior_forms) * topology.boundary[2, 1]).sum(axis=1) * core.Scalar
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
    obtuse_topology, obtuse_vertices = scenarios.obtuse_diamond()
    obtuse_ops = core.Operators(obtuse_topology, obtuse_vertices)
    tails, heads = obtuse_topology.corners[2, 1]
    next_sides, prev_sides = obtuse_ops.sides[tails], obtuse_ops.sides[heads]
    cotangents = -next_sides.dot(prev_sides) / (2 * obtuse_ops.areas)
    unit_amount = core.mv.scalar(np.ones(obtuse_topology.n_elements[2])[..., None]).field().dual()
    cotangent_weights = (~obtuse_topology.incidence[2, 1] * ((cotangents / 2) * unit_amount)).sum(axis=0)
    assert (cotangent_weights.batch()[0] & core.mv.scalar([1.0])).to_array() < -1.0

    obtuse_edge_conductance = core.edge_conductance(obtuse_ops, core.Plane.dual())
    for edge in range(obtuse_topology.n_elements[1]):
        unit_edge = core.mv.scalar((np.arange(obtuse_topology.n_elements[1]) == edge)[..., None]).field()
        assert -(obtuse_edge_conductance(unit_edge) & unit_edge).sites.sum() > 0


def test_isotropic_and_anisotropic_conduction_patch_test_and_flux_balance():
    topology, coordinates = shapes.hexagonal_disk(1.0, 6)
    vertices = scenarios.mv.vector(coordinates).dual().field()
    operators = core.Operators(topology, vertices)
    fibre = (core.mv.xy * (-scenarios.fibre_angle / 2)).exp() >> core.mv.x.dual()
    quarter_turn = (core.mv.xy * (-np.pi / 4)).exp()
    directions = stack((fibre, quarter_turn >> fibre))
    conductivity = (directions * (core.Plane & directions) * scenarios.principal_conductivities).sum(axis=-1)

    # Linear patch test: prescribing a linear potential on the entire boundary recovers it everywhere for any conductivity.
    boundary = topology.exterior[0]
    applied_field = 1.25 * core.mv.x - 0.6 * core.mv.y
    exact_potential = applied_field & vertices
    rim = topology.reading(boundary, 0)
    prescribed_rim = ~rim * (rim * exact_potential)
    potential, current, _ = core.conduction(operators, conductivity, boundary, prescribed_rim)
    np.testing.assert_allclose(potential.batch()[0].kernel, exact_potential.kernel, atol=1e-13)
    np.testing.assert_allclose(potential.batch()[1].kernel, exact_potential.kernel, atol=1e-13)
    expected_current = -conductivity(applied_field)
    np.testing.assert_allclose(current.batch()[0].kernel, np.broadcast_to(expected_current[0].kernel, (topology.n_elements[2], 2)), atol=1e-12)
    np.testing.assert_allclose(current.batch()[1].kernel, np.broadcast_to(expected_current[1].kernel, (topology.n_elements[2], 2)), atol=1e-12)

    # Electrode solve balances flux at every free vertex and conserves total current between the two electrodes.
    abscissa = (core.mv.x & vertices).batch()
    electrodes = boundary[abscissa[boundary].norm() > scenarios.electrode_threshold]
    pinned = topology.reading(electrodes, 0)
    prescribed = ~pinned * (pinned * (core.mv.x & vertices))
    potential, current, streamfunction = core.conduction(operators, conductivity, electrodes, prescribed)

    conductance = operators.gradient.adjugate()(spdiag(operators.areas * conductivity)(operators.gradient))
    free_vertices = np.setdiff1d(np.arange(topology.n_vertices), electrodes)
    free = ~topology.reading(free_vertices, 0) * core.Scalar
    nodal_source = conductance(potential)
    np.testing.assert_allclose(free.adjugate()(nodal_source).kernel, 0.0, atol=1e-11)
    np.testing.assert_allclose(nodal_source.sites.sum().kernel, 0.0, atol=1e-11)
    assert np.isfinite(streamfunction.kernel).all()
