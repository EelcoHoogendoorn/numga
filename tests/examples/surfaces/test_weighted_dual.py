"""A hull and its polar dual about the unit sphere: oriented faces, polar incidence, and the figures."""

import matplotlib.pyplot as plt
import numpy as np

from examples.surfaces.weighted_dual import core, render, scenarios


def test_face_planes_pair_to_one_with_their_vertices_on_a_fixed_mesh():
    frames = [frame for _, frame in zip(range(3), scenarios.breath())]
    assert all(np.array_equal(mesh.faces, frames[0][0].faces) for mesh, _, _ in frames)
    for mesh, points, duals in frames:
        # A face's plane is the dual of its pole.
        assert np.abs((duals.dual_inverse() & (mesh.incidence[2, 0] * points)).to_array() - 1).max() < 1e-10


def test_points_on_the_sphere_give_its_delaunay_mesh_with_poles_as_circumcentres():
    points = scenarios.directions                          # Point-[points]
    mesh = core.hull(points)
    duals = core.poles(mesh, points)
    corners = mesh.incidence[2, 0] * points                                    # [3] Point-[faces]
    centres = duals.normalized()                                               # Point-[faces]
    # Each pole's image is equally far from the three corners of its face: their circumcentre.
    distances = (corners - centres).norm().to_array()                          # [3, faces]
    np.testing.assert_allclose(distances - distances[:1], 0, atol=1e-10)


def test_the_figure_and_two_frames_draw():
    frames = [frame for _, frame in zip(range(2), scenarios.breath())]
    plt.close(render.draw_complex(*frames[0]))
    assert len(list(render.animate(frames))) == 2


def test_the_nearest_rescue_unfolds_the_mesh_steadily():
    settled = list(scenarios.settle())
    mesh = settled[0][0]
    even_weights = core.mv.scalar(np.ones((scenarios.N_POINTS, 1))).field()    # Scalar+[points]
    # The crossed dual edges are exactly the folded ones.
    assert np.array_equal(core.dual_lengths(mesh, settled[0][1]) < 0,
                          core.folds(mesh, scenarios.directions, even_weights) < 0)
    crossed = [(core.dual_lengths(mesh, points) < 0).sum() for _, points, _ in settled]
    assert crossed[0] > 0 and crossed[-1] == 0 and (np.diff(crossed) <= 0).all()
    # Every image on the sphere is held.
    images = [points.normalized() for points in (settled[0][1], settled[-1][1])]   # Point-[points], Point-[points]
    np.testing.assert_allclose((images[1] - images[0]).norm().to_array(), 0, atol=1e-12)
