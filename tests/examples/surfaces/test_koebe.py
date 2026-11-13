"""Koebe polyhedra: the centred circles of the irregular triangulation, carried by one boost, run and draw."""

from examples.surfaces.koebe import render, scenarios


def test_the_boosted_circles_run_and_draw():
    topology, frames = scenarios.boosted(n_frames=1)
    assert len(list(render.animate(topology, frames))) == 1
