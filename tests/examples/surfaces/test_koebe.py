"""Koebe polyhedra: unit circles touching along every edge, under every boost; both polyhedra touch the sphere;
the centred circles balance."""

import numpy as np

from numga import stack
from examples.surfaces.koebe import core, render, scenarios


def centre_distances(graph: core.Mesh, apexes: core.Point) -> core.Scalar:
    """The distance from the centre to the line through the apexes at the ends of each edge of a graph: the line's
    part paired with w, its moment about the centre, over its part wedged with w, its direction."""
    ends = graph.incidence[1, 0] * apexes[..., None]                           # [..., ends] Point-[edges]
    lines = ends[..., 0] & ends[..., 1]                                        # [...] Bivector-[edges]
    return ((lines | core.mv.w).squared() / (lines ^ core.mv.w).squared()).square_root()


def test_circles_touch_along_every_edge_and_both_polyhedra_touch_the_sphere_under_every_boost():
    mesh, frames = scenarios.boosted()
    assert np.abs((frames | frames).to_array() - 1).max() < 1e-10
    ends = mesh.incidence[1, 0] * frames[..., None]                            # [frames, ends] Vector+[edges]
    assert np.abs((ends[..., 0] | ends[..., 1]).to_array() + 1).max() < 1e-10
    assert np.abs(centre_distances(mesh, core.apexes(frames)).to_array() - 1).max() < 1e-10
    duals = core.apexes(core.face_circles(mesh, frames))                       # [frames] Point-[faces]
    # The dual polyhedron's edges join the faces either side of each edge.
    dual_graph = core.Mesh(len(mesh.faces), mesh.edge_faces)
    assert np.abs(centre_distances(dual_graph, duals).to_array() - 1).max() < 1e-10


def test_the_centred_circles_balance_and_the_figure_draws():
    mesh, circles = scenarios.centred()
    touches = core.touching(mesh, circles)                                     # Point-[edges]
    balance = (touches / (core.mv.w & touches)).sites.mean()                   # [] Point-
    assert np.abs((stack((core.mv.x, core.mv.y, core.mv.z)) & balance).to_array()).max() < 1e-4
    assert len(list(render.animate(mesh, circles[None]))) == 1
