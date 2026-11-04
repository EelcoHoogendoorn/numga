"""A lifted planar mesh: tension exactly at convex creases, balanced interior vertices, reciprocal edges across
the bars, and on the paraboloid the Voronoi diagram with the cotangent weights."""

import numpy as np

from examples.surfaces.reciprocal import core, render, scenarios


def test_bars_are_in_tension_at_convex_creases_and_every_interior_vertex_balances():
    topology, points, heights = scenarios.breathing()
    stresses = core.stresses(topology, points, heights)                        # [frames] Scalar+[inner]
    assert np.array_equal(stresses > 0, core.folds(topology, points, heights) > 0)
    assert (stresses < 0).any()
    inner = topology.reading(topology.interior[1], 1)                          # Scalar+[inner <- edges]
    # The bars meeting at a vertex each pull it along themselves by their stress, summed back onto their ends
    # against the way each bar runs.
    bars = (inner * (topology.d[0] * points)).batch()                          # [inner] Point-
    pulls = (bars * stresses).field()                                          # [frames] Point-[inner]
    bar_ends = inner * topology.d[0]                                           # Scalar+[inner <- vertices]
    net = -(~bar_ends * pulls)                                                 # [frames] Point-[vertices]
    assert ((topology.reading(topology.interior[0], 0) * net).norm() < 1e-10).all()
    # The jump in slope across each bar, the faces' planes summed back onto their edges, pairs to nothing with it.
    duals = (core.slopes(topology, points, heights) ^ core.mv.z) * core.mv.z                         # [frames] Plane+[faces]
    assert np.abs(((inner * (~topology.d[1] * duals)).batch() & bars).to_array()).max() < 1e-10


def test_on_the_paraboloid_the_reciprocal_is_the_voronoi_diagram_and_the_stresses_the_cotangent_weights():
    topology, points, heights = scenarios.grid()
    inner = topology.reading(topology.interior[1], 1)                          # Scalar+[inner <- edges]
    edges = topology.boundary[2, 1] * (topology.d[0] * points)                 # [3] Point-[faces]
    tails, heads = topology.corners[2, 1]
    after, before = edges[tails], edges[heads]                                 # [3] Point-[faces], Point-[faces]
    cotangents = -(after.dual_inverse() & before) / after.commutator(before).norm()   # [3] Scalar+[faces]
    weights = (~topology.incidence[2, 1] * (cotangents / 2)).sum(axis=0)       # Scalar+[edges]
    assert np.abs(core.stresses(topology, points, heights).to_array() - (inner * weights).to_array()).max() < 1e-10
    centres = (core.slopes(topology, points, heights) ^ core.mv.z) * core.mv.z                       # Plane+[faces]
    squared_distances = [(centres.dual() - corner).scalar_norm_squared().to_array() for corner in topology.incidence[2, 0] * points]
    assert np.abs(squared_distances[0] - squared_distances[1]).max() < 1e-10 and np.abs(squared_distances[0] - squared_distances[2]).max() < 1e-10
    assert len(list(render.animate(topology, points, [(heights, core.stresses(topology, points, heights),
                                                    (core.slopes(topology, points, heights) ^ core.mv.z) * core.mv.z)]))) == 1
