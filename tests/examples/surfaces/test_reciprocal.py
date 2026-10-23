"""A lifted planar mesh: tension exactly at convex creases, balanced interior vertices, reciprocal edges across
the bars, and on the paraboloid the Voronoi diagram with the cotangent weights."""

import numpy as np

from examples.surfaces.reciprocal import core, render, scenarios


def test_bars_are_in_tension_at_convex_creases_and_every_interior_vertex_balances():
    mesh, heights = scenarios.breathing()
    stresses = core.stresses(mesh, heights)                                    # [frames] Scalar[inner]
    assert np.array_equal(stresses.to_array() > 0, core.folds(mesh, heights).to_array() > 0)
    assert (stresses.to_array() < 0).any()
    inner, left, right = core.beside(mesh)
    # The bars meeting at a vertex each pull it along themselves by their stress.
    pulls = (mesh.edge_displacements.batch()[inner] * stresses).field()        # [frames] Point[inner]
    tails, heads = (mesh.reading(mesh.edges[inner, end]) for end in range(2))  # [inner, vertices] Scalar each
    net = (~tails * pulls - ~heads * pulls).batch()                            # [frames, vertices] Point
    interior = np.setdiff1d(np.arange(len(mesh.vertices.batch())), mesh.edges[mesh.boundary_edges])
    assert net[:, interior].norm().to_array().max() < 1e-10
    duals = core.reciprocal(mesh, heights)                                     # [frames, faces] Vector
    assert np.abs(((duals[:, left] - duals[:, right]) & mesh.edge_displacements.batch()[inner]).to_array()).max() < 1e-10


def test_on_the_paraboloid_the_reciprocal_is_the_voronoi_diagram_and_the_stresses_the_cotangent_weights():
    mesh, heights = scenarios.grid()
    inner, _, _ = core.beside(mesh)
    assert np.abs(core.stresses(mesh, heights).to_array() - mesh.edge_ratio.to_array()[inner]).max() < 1e-10
    centres = core.reciprocal(mesh, heights)                                   # [faces] Vector
    reach = [(centres.dual() - corner.batch()).scalar_norm_squared().to_array() for corner in mesh.corners]
    assert np.abs(reach[0] - reach[1]).max() < 1e-10 and np.abs(reach[0] - reach[2]).max() < 1e-10
    assert len(list(render.animate(mesh, [(heights, core.stresses(mesh, heights), core.reciprocal(mesh, heights))]))) == 1
