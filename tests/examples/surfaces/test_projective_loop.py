"""Vertex weights shape the subdivided surface, and a Loop step keeps unit weights."""

from itertools import islice

import matplotlib.pyplot as plt
import numpy as np

from examples.surfaces.projective_loop import core, render, scenarios


def test_a_heavy_vertex_pulls_the_surface_and_a_light_one_releases_it():
    controls, surfaces, topology = scenarios.weighted()
    heights = (core.mv.z & surfaces.normalized()).batch()                       # [cases, refined] Scalar+
    apex_heights = heights[:, scenarios.TOP_VERTEX]                            # [cases] Scalar+
    assert apex_heights[1] > apex_heights[0] > apex_heights[2]
    # With all weights alike the surface agrees with Loop subdivision of Euclidean positions.
    assert np.all((surfaces[0] & surfaces[0].normalized()).norm() < 1e-12)
    plt.close(render.draw_weights(controls, scenarios.CONTROL_TOPOLOGY, surfaces, topology))
    frames, topology = scenarios.sweep()
    assert len(list(render.animate_weights(islice(frames, 2), scenarios.CONTROL_TOPOLOGY, topology))) == 2


def test_a_loop_step_keeps_unit_weights_and_splits_every_face_in_four():
    topology = scenarios.CONTROL_TOPOLOGY.loop_subdivided(np.zeros(len(scenarios.OCTAHEDRON), dtype=bool), 1)
    points = topology.from_coarser * core.mv.point(scenarios.OCTAHEDRON).field()
    np.testing.assert_allclose((core.mv.w & points).batch().kernel[..., 0], 1, atol=1e-12)
    assert topology.n_elements[2] == 4 * len(scenarios.FACES)
    # Every edge of the refined mesh lies between two of its faces, once in each direction.
    face_ones = core.mv.scalar(np.ones((topology.n_elements[2], 1))).field()
    np.testing.assert_array_equal((~topology.incidence[2, 1].sum(axis=0) * face_ones).to_array(), 2)
    np.testing.assert_array_equal((~topology.d[1] * face_ones).to_array(), 0)
