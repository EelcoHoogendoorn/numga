"""Centred Koebe circles solve to external tangency, stay tangent under a boost, and render both static and animated views."""

import matplotlib.pyplot as plt

from examples.surfaces.koebe import core, render, scenarios


def test_koebe_circles_solve_boost_and_render():
    topology, circles = scenarios.centred_polyhedron()
    figure = render.polyhedron_figure(topology, circles)
    plt.close(figure)

    boosted = next(core.boost_states(circles, scenarios.rapidity, scenarios.tilt, 2))
    assert ((boosted.squared() - 1).abs() < 1e-10).all()
    ends = topology.incidence[1, 0] * boosted                               # [2] Plane+[E]
    assert (((ends[0] | ends[1]) + 1).abs() < 1e-10).all()

    view = render.polyhedron_view(core.cone_apexes(boosted), 80)
    frame = next(render.boost_animation(topology, view, [boosted]))
    assert frame.ndim == 3 and frame.shape[2] == 3
