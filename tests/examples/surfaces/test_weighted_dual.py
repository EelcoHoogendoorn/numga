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
    corners = render.coordinates(points)[faces]                                # [faces, 3, 3]
    centres = render.coordinates(duals)
    centres = centres / np.linalg.norm(centres, axis=-1, keepdims=True)
    # Each pole's image is equally far from the three corners of its face: their circumcentre.
    distances = np.linalg.norm(corners - centres[:, None], axis=-1)
    np.testing.assert_allclose(distances - distances[:, :1], 0, atol=1e-10)


def test_the_figure_and_two_frames_draw():
    frames = [frame for _, frame in zip(range(2), scenarios.breath())]
    plt.close(render.draw_complex(*frames[0]))
    assert len(list(render.animate(frames))) == 2


def test_the_nearest_rescue_unfolds_the_mesh_steadily():
    settled = list(scenarios.settle())
    faces = settled[0][1]
    # The crossed dual edges are exactly the folded ones.
    assert np.array_equal(scenarios.scalars(core.dual_lengths(settled[0][0], faces)) < 0,
                          scenarios.scalars(core.folds(scenarios.directions(scenarios.N_POINTS), np.ones(scenarios.N_POINTS), faces)) < 0)
    crossed = [(scenarios.scalars(core.dual_lengths(points, faces)) < 0).sum() for points, _, _ in settled]
    assert crossed[0] > 0 and crossed[-1] == 0 and (np.diff(crossed) <= 0).all()
    # Every image on the sphere is held.
    images = [render.coordinates(points) / np.linalg.norm(render.coordinates(points), axis=-1, keepdims=True) for points in (settled[0][0], settled[-1][0])]
    np.testing.assert_allclose(images[0], images[1], atol=1e-12)


def flip(faces: np.ndarray, edge: int) -> np.ndarray:
    """The mesh with one edge turned to join the other two corners of the faces beside it."""
    edges, beside = core.adjacency(faces)
    first, second = beside[edge]
    a, b = edges[edge]
    c, d = faces[first].sum() - a - b, faces[second].sum() - a - b
    turned = faces[first][(list(faces[first]).index(a) + 1) % 3] != b
    a, b = (b, a) if turned else (a, b)
    flipped = faces.copy()
    flipped[first], flipped[second] = (a, d, c), (d, b, c)
    return flipped


def test_a_mesh_that_is_no_hull_at_any_distances_has_no_rescue():
    # The sphere's mesh with so many random edges flipped, from this seed.
    flips, seed = 8, 6
    rng = np.random.default_rng(seed)
    faces = core.hull(scenarios.directions(scenarios.N_POINTS))
    for _ in range(flips):
        faces = flip(faces, rng.integers(len(core.adjacency(faces)[0])))
    assert scenarios.widest(scenarios.fold_map(faces))[1] < 0
