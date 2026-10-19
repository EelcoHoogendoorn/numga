"""A lifted planar mesh: tension exactly at convex creases, balanced interior vertices, reciprocal edges across
the bars, and on the paraboloid the Voronoi diagram with the cotangent weights."""

import numpy as np

from examples.surfaces.reciprocal import core, render, scenarios


def test_bars_are_in_tension_at_convex_creases_and_every_interior_vertex_balances():
    mesh, heights = scenarios.breathing()
    stresses = core.stresses(mesh, heights)                                    # [frames] Scalar[inner]
    assert np.array_equal(render.scalars(stresses) > 0, render.scalars(core.folds(mesh, heights)) > 0)
    assert (render.scalars(stresses) < 0).any()
    assert render.unbalanced(mesh, stresses) < 1e-10
    inner, left, right = core.beside(mesh)
    duals = core.reciprocal(mesh, heights)                                     # [frames, faces] Vector
    assert np.abs(render.scalars((duals[:, left] - duals[:, right]) & mesh.edge_displacements.batch()[inner])).max() < 1e-10


def test_on_the_paraboloid_the_reciprocal_is_the_voronoi_diagram_and_the_stresses_the_cotangent_weights():
    mesh, heights = scenarios.grid()
    inner, _, _ = core.beside(mesh)
    assert np.abs(render.scalars(core.stresses(mesh, heights)) - render.scalars(mesh.edge_ratio)[inner]).max() < 1e-10
    centres = core.reciprocal(mesh, heights)                                   # [faces] Vector
    reach = [render.scalars((centres.dual() - corner.batch()).scalar_norm_squared()) for corner in mesh.corners]
    assert np.abs(reach[0] - reach[1]).max() < 1e-10 and np.abs(reach[0] - reach[2]).max() < 1e-10
    assert len(render.animate(mesh, [(heights, core.stresses(mesh, heights), core.reciprocal(mesh, heights))])) == 1
