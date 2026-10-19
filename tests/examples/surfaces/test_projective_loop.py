"""Loop subdivision of weighted points: weights shape the surface, and refining commutes with perspective."""

from itertools import islice

import matplotlib.pyplot as plt
import numpy as np

from numga import stack
from examples.surfaces.projective_loop import core, render, scenarios


def test_refining_commutes_with_perspective_for_weighted_points_only():
    seen, weighted_orders, positional_orders, faces = scenarios.perspective()
    assert render.separation(weighted_orders) < 1e-12
    assert render.separation(positional_orders) > 1e-2
    plt.close(render.draw_orders(seen, scenarios.FACES, weighted_orders, positional_orders, faces))


def test_a_heavy_vertex_pulls_the_surface_and_a_light_one_releases_it():
    controls, surfaces, faces = scenarios.weighted()
    top = render.coordinates(surfaces)[..., 2].max(axis=-1)                   # [cases]
    assert top[1] > top[0] > top[2]
    # With all weights alike the surface is Loop subdivision of the positions.
    assert render.separation(stack([surfaces[0], core.unit(surfaces[0])])) < 1e-12
    plt.close(render.draw_weights(controls, scenarios.FACES, surfaces, faces))
    frames, faces = scenarios.sweep()
    assert len(render.animate_weights(islice(frames, 2), scenarios.FACES, faces)) == 2


def test_a_loop_step_keeps_unit_weights_and_splits_every_face_in_four():
    finer = core.Mesh(core.mv.point(scenarios.OCTAHEDRON).field(), scenarios.FACES).subdivide_loop(np.zeros(len(scenarios.OCTAHEDRON), dtype=bool))
    np.testing.assert_allclose((core.mv.w & finer.vertices).batch().kernel[..., 0], 1, atol=1e-12)
    assert len(finer.faces) == 4 * len(scenarios.FACES)
    # Every edge of the finer mesh lies between two of its faces, once each way round.
    sides = np.stack([finer.faces, np.roll(finer.faces, -1, axis=1)], axis=-1).reshape(-1, 2)
    assert len(np.unique(sides, axis=0)) == len(sides) == 2 * len(finer.edges)
