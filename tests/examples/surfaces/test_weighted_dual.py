"""A hull and its polar dual about the unit sphere: oriented faces and polar incidence; the figure
and a frame of the breath and of the rescue draw."""

from itertools import islice

import matplotlib.pyplot as plt
import numpy as np

from examples.surfaces.weighted_dual import core, render, scenarios


def test_face_planes_pair_to_one_with_their_vertices_on_a_fixed_mesh():
    frames = [frame for _, frame in zip(range(3), scenarios.breath())]
    assert all(np.array_equal(topology.elements[2], frames[0][0].elements[2]) for topology, _, _ in frames)
    for topology, points, duals in frames:
        # A face's plane is the dual of its pole.
        assert np.abs((duals.dual_inverse() & (topology.incidence[2, 0] * points)).to_array() - 1).max() < 1e-10


def test_points_on_the_sphere_give_its_delaunay_mesh_with_poles_as_circumcentres():
    points = scenarios.directions                          # Point-[points]
    topology = core.hull(points)
    duals = core.poles(topology, points)
    corners = topology.incidence[2, 0] * points                                # [3] Point-[faces]
    centres = duals.normalized()                                               # Point-[faces]
    # Each pole's image is equally far from the three corners of its face: their circumcentre.
    distances = (corners - centres).norm().to_array()                          # [3, faces]
    np.testing.assert_allclose(distances - distances[:1], 0, atol=1e-10)


def test_the_figure_and_a_frame_of_each_animation_draw():
    frames = list(islice(scenarios.breath(), 1))
    plt.close(render.draw_complex(*frames[0]))
    assert len(list(render.animate(frames))) == 1
    assert len(list(render.animate(islice(scenarios.settle(), 1)))) == 1
