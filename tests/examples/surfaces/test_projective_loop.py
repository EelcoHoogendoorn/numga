"""Vertex weights shape the subdivided surface, and a Loop step preserves unit weights."""

from itertools import islice

import matplotlib.pyplot as plt
import numpy as np

from examples.surfaces.projective_loop import core, render, scenarios


def test_a_heavy_vertex_pulls_the_surface_and_a_light_one_releases_it():
    controls, surfaces, topology = scenarios.weighted_cases()
    heights = (core.mv.z & surfaces.normalized()).batch()                   # [3, V_refined] Scalar+
    apex_heights = heights[:, scenarios.top_vertex]                         # [3] Scalar+
    assert apex_heights[1] > apex_heights[0] > apex_heights[2]
    # With all weights alike the surface agrees with Loop subdivision of Euclidean positions.
    assert ((surfaces[0] & surfaces[0].normalized()).norm() < 1e-12).all()
    plt.close(render.weights_figure(controls, scenarios.control_topology, surfaces, topology))
    frames, topology = scenarios.weight_sweep()
    assert len(list(render.weights_animation(islice(frames, 2), scenarios.control_topology, topology))) == 2
    control, weighted_orders, positional_orders, topology = scenarios.perspective_orders()
    weighted_unit = weighted_orders.normalized()                            # [2] Point-[V_refined]
    assert ((weighted_unit[0] & weighted_unit[1]).norm() < 1e-12).all()
    plt.close(render.orders_figure(control, scenarios.control_topology, weighted_orders, positional_orders, topology))


def test_a_loop_step_keeps_unit_weights_and_splits_every_face_in_four():
    V_octahedron, E_octahedron, F_octahedron = scenarios.control_topology.n_elements
    topology = scenarios.control_topology.loop_subdivided(np.zeros(V_octahedron, dtype=bool), 1)
    V, E, F = topology.n_elements
    points = topology.from_coarser * core.mv.point(scenarios.octahedron).field()  # [] Point-[V]
    assert (((core.mv.w & points) - 1).abs() < 1e-12).all()
    assert F == 4 * F_octahedron
    # Every edge of the refined topology lies between two of its faces, once in each direction.
    face_ones = core.mv.scalar(np.ones((F, 1))).field()                     # [] Scalar+[F]
    assert ((topology.incidence[2, 1].sum(axis=0).adjugate() * face_ones - 2).abs() < 1e-12).all()
    assert ((topology.d[1].adjugate() * face_ones).abs() < 1e-12).all()
