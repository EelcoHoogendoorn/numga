"""A hull and its polar dual about the unit sphere: oriented faces, polar incidence, and the figures."""

import matplotlib.pyplot as plt
import numpy as np

from examples.surfaces.weighted_dual import core, render, scenarios


def test_face_planes_pair_to_one_with_their_vertices_on_a_fixed_mesh():
    frames = [frame for _, frame in zip(range(3), scenarios.breath())]
    assert all(np.array_equal(faces, frames[0][1]) for _, faces, _ in frames)
    for points, faces, duals in frames:
        # A face's plane is the dual of its pole.
        assert np.abs((duals.dual_inverse()[:, None] & points[faces]).to_array() - 1).max() < 1e-10


def test_points_on_the_sphere_give_its_delaunay_mesh_with_poles_as_circumcentres():
    points = scenarios.directions(scenarios.N_POINTS)                          # [points] Point
    faces = core.hull(points)
    duals = core.poles(points, faces)
    corners = points[faces]                                                    # [faces, 3] Point
    centres = duals.normalized()                                               # [faces] Point
    # Each pole's image is equally far from the three corners of its face: their circumcentre.
    distances = (corners - centres[:, None]).norm().to_array()                 # [faces, 3]
    np.testing.assert_allclose(distances - distances[:, :1], 0, atol=1e-10)


def test_the_figure_and_two_frames_draw():
    frames = [frame for _, frame in zip(range(2), scenarios.breath())]
    plt.close(render.draw_complex(*frames[0]))
    assert len(list(render.animate(frames))) == 2


def test_the_nearest_rescue_unfolds_the_mesh_steadily():
    settled = list(scenarios.settle())
    faces = settled[0][1]
    # The crossed dual edges are exactly the folded ones.
    assert np.array_equal(core.dual_lengths(settled[0][0], faces).to_array() < 0,
                          core.folds(scenarios.directions(scenarios.N_POINTS), np.ones(scenarios.N_POINTS), faces).to_array() < 0)
    crossed = [(core.dual_lengths(points, faces).to_array() < 0).sum() for points, _, _ in settled]
    assert crossed[0] > 0 and crossed[-1] == 0 and (np.diff(crossed) <= 0).all()
    # Every image on the sphere is held.
    images = [points.normalized() for points in (settled[0][0], settled[-1][0])]   # [points] Point each
    np.testing.assert_allclose((images[1] - images[0]).norm().to_array(), 0, atol=1e-12)
