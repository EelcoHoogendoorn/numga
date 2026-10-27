"""A lifted planar mesh: tension exactly at convex creases, balanced interior vertices, reciprocal edges across
the bars, and on the paraboloid the Voronoi diagram with the cotangent weights."""

import numpy as np

from examples.surfaces.reciprocal import core, render, scenarios


def test_bars_are_in_tension_at_convex_creases_and_every_interior_vertex_balances():
    mesh, points, heights = scenarios.breathing()
    stresses = core.stresses(mesh, points, heights)                            # [frames] Scalar+[inner]
    assert np.array_equal(stresses > 0, core.folds(mesh, points, heights) > 0)
    assert (stresses < 0).any()
    inner, (left, right) = mesh.interior_edges, mesh.edge_faces.T
    # The bars meeting at a vertex each pull it along themselves by their stress.
    bars = (mesh.d0 * points).batch()[inner]                                   # [inner] Point-
    pulls = (bars * stresses).field()                                          # [frames] Point-[inner]
    tails, heads = (mesh.reading(mesh.edges[inner, end]) for end in range(2))  # Scalar+[inner <- vertices], Scalar+[inner <- vertices]
    net = (~tails * pulls - ~heads * pulls).batch()                            # [frames, vertices] Point-
    interior = np.setdiff1d(np.arange(mesh.n_vertices), mesh.edges[mesh.boundary_edges])
    assert (net[:, interior].norm() < 1e-10).all()
    duals = (core.slopes(mesh, points, heights) ^ core.mv.z) * core.mv.z                             # [frames, faces] Plane+
    assert np.abs(((duals[:, left] - duals[:, right]) & bars).to_array()).max() < 1e-10


def test_on_the_paraboloid_the_reciprocal_is_the_voronoi_diagram_and_the_stresses_the_cotangent_weights():
    mesh, points, heights = scenarios.grid()
    inner = mesh.interior_edges
    edges = ((mesh.d0 * points).batch()[mesh.face_edges.T] * mesh.face_edge_orientation.T).field()   # [3] Point-[faces]
    after, before = edges[[1, 2, 0]], edges[[2, 0, 1]]                         # [3] Point-[faces], Point-[faces]
    cotangents = -(after.dual_inverse() & before) / after.commutator(before).norm()   # [3] Scalar+[faces]
    weights = ~mesh.by_corner(cotangents / 2) * core.mesh_module.as_scalar(np.ones(len(mesh.faces))).field()   # Scalar+[edges]
    assert np.abs(core.stresses(mesh, points, heights).to_array() - weights.to_array()[inner]).max() < 1e-10
    centres = (core.slopes(mesh, points, heights) ^ core.mv.z) * core.mv.z                           # [faces] Plane+
    squared_distances = [(centres.dual() - corner).scalar_norm_squared().to_array() for corner in points.batch()[mesh.faces.T]]
    assert np.abs(squared_distances[0] - squared_distances[1]).max() < 1e-10 and np.abs(squared_distances[0] - squared_distances[2]).max() < 1e-10
    assert len(list(render.animate(mesh, points, [(heights, core.stresses(mesh, points, heights),
                                                    (core.slopes(mesh, points, heights) ^ core.mv.z) * core.mv.z)]))) == 1
