"""Simplex gradients, vertex and edge Whitney forms, tetrahedral forms, midpoint quadrature, and steady conduction satisfy exactness and conservation checks."""

from functools import reduce
from operator import and_

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from numga import stack
from numga.sparse import spdiag
from numga.topology import simplex_parity
from examples.surfaces.whitney_forms import core, render, scenarios
from examples.util import shapes


def test_vertex_and_edge_whitney_forms_on_triangle_and_diamond():
    triangle, vertices, points = scenarios.single_triangle(4)
    corners = triangle.incidence[2, 0] * vertices                            # [3] Point-[F]
    patches, gradients = core.simplex_gradients(triangle, vertices)          # Patch-[F], [3] Plane+[F]
    vertex_forms = core.vertex_whitney_forms(triangle, vertices, points)     # [3, ...] Scalar+[F]
    corner_values = core.vertex_whitney_forms(triangle, vertices, corners)   # [3, 3] Scalar+[F]

    assert vertex_forms.batch().gatype == core.Scalar
    assert gradients.batch().gatype == core.Plane
    assert ((corner_values - np.eye(3)).abs() < 1e-10).all()
    assert ((vertex_forms.sum(axis=0) - 1).abs() < 1e-10).all()
    assert (gradients.sum(axis=0).norm() < 1e-10).all()

    # Affine functions and their gradients are reproduced exactly from vertex values.
    true_gradient = 1.4 * core.mv.x - 0.85 * core.mv.y
    prescribed_corners = 0.6 + (true_gradient & corners)
    prescribed_samples = 0.6 + (true_gradient & points)
    interpolated = (prescribed_corners[:, None] * vertex_forms).sum(axis=0)
    reconstructed_gradient = (prescribed_corners * gradients).sum(axis=0)
    assert ((interpolated - prescribed_samples).abs() < 1e-10).all()
    assert ((reconstructed_gradient - true_gradient).norm() < 1e-10).all()

    edge_forms = core.edge_whitney_forms(triangle, vertices, points)         # [3, ...] Plane+[F]
    tails, heads = triangle.corners[2, 1]
    alternating_forms = triangle.orientation[1][0, :, None] * (
        vertex_forms[tails] * gradients[heads, None] - vertex_forms[heads] * gradients[tails, None]
    )
    assert ((edge_forms - alternating_forms).norm() < 1e-10).all()

    sides = triangle.boundary[2, 1] * (triangle.d[0] * vertices)             # [3] Point-[F]
    first, second = np.where(triangle.orientation[1] > 0, triangle.complements[2, 0], triangle.complements[2, 0][::-1])
    edge_points = corners[first] * 0.35 + corners[second] * 0.65
    readings = core.edge_whitney_forms(triangle, vertices, edge_points) & sides[None, :]
    assert ((readings - np.eye(3)).abs() < 1e-10).all()

    diamond, diamond_vertices = scenarios.edge_diamond()
    V_diamond, E_diamond, F_diamond = diamond.n_elements
    diamond_edges = diamond.d[0] * diamond_vertices
    edge_samples = ((diamond.incidence[1, 0] * np.array([[0.15], [0.85]])).sum(axis=0)) * diamond_vertices
    shared_reading = diamond.reading(diamond.interior[1], 1)                 # Scalar+[1 <- E_diamond]
    face_from_shared = (diamond.incidence[2, 1] * shared_reading.adjugate()).sum(axis=0)
    seam_points = face_from_shared * (shared_reading * edge_samples)
    seam_edges = face_from_shared * (shared_reading * diamond_edges)
    shared_circulation = shared_reading.adjugate() * core.mv.scalar([[1]]).field()  # Scalar+[E_diamond]
    seams = core.edge_whitney_map(diamond, diamond_vertices, seam_points[None])[0](shared_circulation)
    assert (((seams & seam_edges) - 1).abs() < 1e-10).all()
    normal_readings = (seams & core.mv.x.dual()).batch()
    assert ((normal_readings[0] - normal_readings[1]).abs() > 0.5).all()

    star_topology, star_coordinates = shapes.hexagonal_disk(1.0, 1)
    V_star, E_star, F_star = star_topology.n_elements
    star_vertices = core.mv.vector(star_coordinates).dual().field()          # Point-[V_star]
    vertex_form = core.mv.scalar((np.arange(V_star) == 0)[..., None]).field()  # Scalar+[V_star]
    fig0 = render.vertex_forms_figure(triangle, vertices, gradients, star_topology, star_vertices, vertex_form)
    assert isinstance(fig0, Figure)
    plt.close(fig0)

    corner_forms = core.edge_whitney_forms(triangle, vertices, corners)
    diamond_corners = diamond.incidence[2, 0] * diamond_vertices
    shared_corner_forms = core.edge_whitney_map(diamond, diamond_vertices, diamond_corners)(shared_circulation)
    fig1 = render.edge_forms_figure(triangle, vertices, corner_forms, diamond, diamond_vertices, shared_corner_forms)
    assert isinstance(fig1, Figure)
    plt.close(fig1)


def test_tetrahedral_whitney_forms_helicity_and_render():
    topology, vertices, points = scenarios.single_tetrahedron(3)
    forms = [core.tetrahedral_forms(topology, vertices, degree) for degree in range(3)]
    vertex_forms, edge_forms, face_forms = forms
    weights = vertex_forms[:, None] & points[None, :]

    assert vertex_forms.batch().gatype == core.space.gatype.vector()
    assert weights.batch().gatype == core.SpaceScalar
    assert (edge_forms[:, None] & points[None, :]).batch().gatype == core.space.gatype.vector()
    assert (face_forms[:, None] & points[None, :]).batch().gatype.refines(core.space.gatype.bivector())

    for degree, degree_forms in enumerate(forms):
        spans = reduce(and_, topology.on[3, degree] * vertices)
        assert (((degree_forms[:, None] & spans[None, :]) - np.eye(spans.shape[0])).abs() < 1e-10).all()
    assert ((weights.sum(axis=0) - 1).abs() < 1e-10).all()

    corners = topology.incidence[3, 0] * vertices                            # [4] Point-[T]
    tails, heads = topology.corners[3, 1]
    edge_spans = reduce(and_, topology.on[3, 1] * vertices)
    assert (((edge_forms - (vertex_forms[tails] ^ vertex_forms[heads]))[:, None] & edge_spans[None, :]).abs() < 1e-10).all()
    first, second, third = topology.corners[3, 2]
    face_spans = reduce(and_, topology.on[3, 2] * vertices)
    assert (((face_forms - (vertex_forms[first] ^ vertex_forms[second] ^ vertex_forms[third]))[:, None] & face_spans[None, :]).abs() < 1e-10).all()

    tet_helicity = core.tetrahedral_helicity(topology, vertices)
    deformed_helicity = core.tetrahedral_helicity(topology, scenarios.deformed_tetrahedron())
    assert tet_helicity.batch().gatype == core.SpaceScalar
    assert ((tet_helicity - deformed_helicity).abs() < 1e-10).all()
    edge_corners = topology.corners[3, 1].T
    edge_pairs = np.concatenate(np.broadcast_arrays(edge_corners[:, None], edge_corners[None, :]), axis=-1)
    disjoint = (edge_corners[:, None, :, None] != edge_corners[None, :, None, :]).all(axis=(-1, -2))
    linking = np.where(disjoint, simplex_parity(edge_pairs), 0) / 6
    assert ((tet_helicity - linking).abs() < 1e-10).all()

    pages = core.tetrahedral_edge_pages(topology, vertices, 2)
    lines = core.tetrahedral_face_lines(topology, vertices, scenarios.face_part_middles(2))
    fig = render.tetrahedron_forms_figure(topology, vertices, pages, lines)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_cochain_commutation_midpoint_quadrature_and_conduction():
    topology, coordinates = shapes.hexagonal_disk(1.0, 2)
    V, E, F = topology.n_elements
    vertices = scenarios.mv.vector(coordinates).dual().field()               # Point-[V]
    d0 = topology.d[0] * core.Scalar                                         # (Scalar+ <- Scalar+)[E <- V]
    d1 = topology.d[1] * core.Scalar                                         # (Scalar+ <- Scalar+)[F <- E]
    patches, corner_gradients = core.simplex_gradients(topology, vertices)   # Patch-[F], [3] Plane+[F]
    gradient = core.simplex_gradient_map(topology, vertices)                 # (Plane+ <- Scalar+)[F <- V]
    whitney1, curl, whitney2 = core.cochain_maps(topology, vertices)

    potential_sample = (core.mv.x & vertices) * 0.9 - (core.mv.y & vertices) * 1.3
    assert (d1(d0(potential_sample)).abs() < 1e-10).all()
    assert ((whitney1(d0(potential_sample)) - gradient(potential_sample)).norm() < 1e-10).all()
    cochain = d0(potential_sample) + core.mv.scalar(np.cos(np.arange(E))[..., None]).field()
    assert ((curl(cochain) - whitney2(d1(cochain))).norm() < 1e-10).all()
    face_cochain = d1(cochain)
    assert (((whitney2(face_cochain) & patches) - face_cochain).abs() < 1e-10).all()

    fibre_turn = (core.mv.xy * (-scenarios.fibre_angle / 2)).exp()           # [] Rotor+
    fibre = fibre_turn >> core.mv.x.dual()                                   # [] Point-
    quarter_turn = (core.mv.xy * (-np.pi / 4)).exp()                         # [] Rotor+
    directions = stack((fibre, quarter_turn >> fibre))                       # [2] Point-
    conductivity = (directions * (core.Plane & directions) * scenarios.principal_conductivities).sum(axis=-1)  # [2] Point- <- Plane+

    edge_conductances = core.edge_conductance(topology, vertices, conductivity)
    interior_weights = np.array([[2 / 3, 1 / 6, 1 / 6], [1 / 6, 2 / 3, 1 / 6], [1 / 6, 1 / 6, 2 / 3]])
    corners = topology.incidence[2, 0] * vertices                            # [3] Point-[F]
    interior_points = (corners[None, :] * interior_weights).sum(axis=1)      # [3] Point-[F]
    interior_map = core.edge_whitney_map(topology, vertices, interior_points)
    areas = patches.antinorm()                                               # Scalar+[F]
    quadrature_weights = ((areas / 3) * conductivity)[..., None]             # [2, 3] (Point- <- Plane+)[F]
    interior_conductances = interior_map.adjugate()(quadrature_weights(interior_map)).sum(axis=-1)
    assert ((edge_conductances(cochain) - interior_conductances(cochain)).antinorm() < 1e-10).all()

    conductance = gradient.adjugate()((areas * conductivity)(gradient))
    quadratic = (core.mv.x & vertices).squared() - 0.4 * (core.mv.y & vertices).squared()
    assert ((d0.adjugate()(edge_conductances(d0(quadratic))) - conductance(quadratic)).antinorm() < 1e-10).all()

    obtuse_topology, obtuse_vertices = scenarios.obtuse_diamond()
    V_obtuse, E_obtuse, F_obtuse = obtuse_topology.n_elements
    obtuse_conductance = core.edge_conductance(obtuse_topology, obtuse_vertices, core.Plane.dual())
    for edge_index in range(E_obtuse):
        unit_edge = core.mv.scalar((np.arange(E_obtuse) == edge_index)[..., None]).field()
        assert ((unit_edge & obtuse_conductance(unit_edge)).sites.sum() > 0).all()

    boundary = topology.exterior[0]
    applied_field = 1.25 * core.mv.x - 0.6 * core.mv.y
    exact_potential = applied_field & vertices
    rim = topology.reading(boundary, 0)
    prescribed_rim = rim.adjugate() * (rim * exact_potential)
    patched_potential, patched_current = core.steady_conduction(topology, vertices, conductivity, boundary, prescribed_rim)
    assert ((patched_potential - exact_potential).abs() < 1e-10).all()
    expected_current = -conductivity(applied_field)
    assert ((patched_current.batch() - expected_current[:, None]).antinorm() < 1e-10).all()

    exterior = topology.reading(boundary, 0)
    electrodes = boundary[(exterior * (core.mv.x & vertices)).abs() > scenarios.electrode_threshold]
    pinned = topology.reading(electrodes, 0)
    prescribed = pinned.adjugate() * (pinned * (core.mv.x & vertices))
    potential, current = core.steady_conduction(topology, vertices, conductivity, electrodes, prescribed)
    nodal_source = conductance(potential)
    free_vertices = np.setdiff1d(np.arange(V), electrodes)
    free = topology.reading(free_vertices, 0).adjugate() * core.Scalar
    assert (free.adjugate()(nodal_source).antinorm() < 1e-9).all()
    assert (nodal_source.sites.sum().antinorm() < 1e-9).all()

    fig = render.conduction_figure(topology, vertices, potential, current, fibre)
    assert isinstance(fig, Figure)
    plt.close(fig)
