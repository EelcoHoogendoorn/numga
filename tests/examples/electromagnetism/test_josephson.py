"""Coarse triangulated grids exercise the Josephson junction operators, fluxon scenarios, and renderers."""

import matplotlib.pyplot as plt

from examples.electromagnetism.josephson import core, render, scenarios


def test_josephson_scenarios_and_render():
    pivots, pendulum_stream = scenarios.pendulum_collision(n_columns=41)
    first_pendula = next(pendulum_stream)
    assert ((first_pendula.norm() - 1).abs() < 1e-12).all()
    frame = next(render.pendulum_frames(pivots, [first_pendula]))
    assert frame.ndim == 3 and frame.shape[-1] == 3

    speeds, initial_strengths, final_strengths = scenarios.lorentz_contraction(n_columns=41, n_steps=1)
    assert (initial_strengths.batch() >= 0).all()
    plt.close(render.contraction_figure(speeds, initial_strengths, final_strengths))

    ring_topology, ring_vertices, ring_stream = scenarios.collapsing_ring(n_side=9)
    areas, _ = core.simplex_gradients(ring_topology, ring_vertices)
    assert (areas.batch() > 0).all()
    next(render.current_frames(ring_topology, ring_vertices, ring_stream))

    wide_topology, wide_vertices, line_stream = scenarios.edge_reflection(n_columns=9, n_rows=5)
    next(render.current_frames(wide_topology, wide_vertices, line_stream))
