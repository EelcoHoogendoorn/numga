"""Koebe polyhedra: unit circles touching along every edge, under every boost; both polyhedra touch the sphere;
the centred circles balance."""

import numpy as np

from examples.surfaces.koebe import core, render, scenarios


def test_circles_touch_along_every_edge_and_both_polyhedra_touch_the_sphere_under_every_boost():
    frames, faces, edges = scenarios.boosted()
    assert np.abs((frames | frames).to_array() - 1).max() < 1e-10
    assert np.abs((frames[:, edges[:, 0]] | frames[:, edges[:, 1]]).to_array() + 1).max() < 1e-10
    assert np.abs(render.edge_distances(core.apexes(frames), edges) - 1).max() < 1e-10
    duals = core.apexes(core.face_circles(frames, faces))                      # [frames, faces] Point
    assert np.abs(render.edge_distances(duals, render.beside(faces, edges)) - 1).max() < 1e-10


def test_the_centred_circles_balance_and_the_figure_draws():
    circles, faces, edges = scenarios.centred()
    assert np.linalg.norm(render.coordinates(core.touching(circles, edges).dual()).mean(axis=0)) < 1e-4
    assert len(list(render.animate(circles[None], faces, edges))) == 1
