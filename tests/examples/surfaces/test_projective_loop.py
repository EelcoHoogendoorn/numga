"""Vertex weights shape the subdivided surface, and refining commutes with perspective for weighted points."""

from itertools import islice

import matplotlib.pyplot as plt
import numpy as np

from examples.surfaces.projective_loop import core, render, scenarios


def test_refining_commutes_with_perspective_for_weighted_points_only():
    control, weighted_orders, positional_orders, refined_mesh = scenarios.perspective()
    weighted_unit = weighted_orders.normalized()
    positional_unit = positional_orders.normalized()
    assert np.all((weighted_unit[0] & weighted_unit[1]).norm() < 1e-12)
    assert np.any((positional_unit[0] & positional_unit[1]).norm() > 1e-2)
    plt.close(render.draw_orders(control, scenarios.CONTROL_MESH, weighted_orders, positional_orders, refined_mesh))


def test_a_heavy_vertex_pulls_the_surface_and_a_light_one_releases_it():
    controls, surfaces, refined_mesh = scenarios.weighted()
    heights = (core.mv.z & surfaces.normalized()).batch()                       # [cases, refined] Scalar+
    apex_heights = heights[:, scenarios.TOP_VERTEX]                            # [cases] Scalar+
    assert apex_heights[1] > apex_heights[0] > apex_heights[2]
    # With all weights alike the surface agrees with Loop subdivision of Euclidean positions.
    assert np.all((surfaces[0] & surfaces[0].normalized()).norm() < 1e-12)
    plt.close(render.draw_weights(controls, scenarios.CONTROL_MESH, surfaces, refined_mesh))
    frames, refined_mesh = scenarios.sweep()
    assert len(list(render.animate_weights(islice(frames, 2), scenarios.CONTROL_MESH, refined_mesh))) == 2


def test_a_loop_step_keeps_unit_weights_and_splits_every_face_in_four():
    refined_mesh = scenarios.CONTROL_MESH.loop_subdivided(np.zeros(len(scenarios.OCTAHEDRON), dtype=bool), 1)
    points = refined_mesh.from_coarser * core.mv.point(scenarios.OCTAHEDRON).field()
    np.testing.assert_allclose((core.mv.w & points).batch().kernel[..., 0], 1, atol=1e-12)
    assert len(refined_mesh.faces) == 4 * len(scenarios.FACES)
    # Every edge of the refined mesh lies between two of its faces, once in each direction.
    face_ones = core.mv.scalar(np.ones((len(refined_mesh.faces), 1))).field()
    np.testing.assert_array_equal((~refined_mesh.incidence[2, 1].sum(axis=0) * face_ones).to_array(), 2)
    np.testing.assert_array_equal((~refined_mesh.d[1] * face_ones).to_array(), 0)
