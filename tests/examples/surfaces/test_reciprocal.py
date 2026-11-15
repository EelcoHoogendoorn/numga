"""Coverage test for the lifted triangulation, bar stresses, and reciprocal diagram."""

import matplotlib.pyplot as plt

from examples.surfaces.reciprocal import core, render, scenarios


def test_one_breath_runs_and_draws():
    topology, points, paraboloid = scenarios.paraboloid_grid()
    stresses = core.bar_stresses(topology, points, paraboloid)
    duals = (core.face_planes(topology, points, paraboloid) ^ core.mv.z) * core.mv.z
    figure = render.reciprocal_figure(topology, points, paraboloid, stresses, duals)
    plt.close(figure)
    phases = scenarios.vertex_phases(topology)
    states = core.breathing_states(topology, points, paraboloid, phases, scenarios.amplitude, 1)
    assert len(list(render.breath_animation(topology, points, states))) == 1
