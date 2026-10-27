"""Koebe polyhedra: unit circles touching along every edge, under every boost; both polyhedra touch the sphere;
the centred circles balance."""

import numpy as np

from examples.surfaces.koebe import core, render, scenarios


def centre_distances(apexes: core.Point, edges: np.ndarray) -> core.Scalar:
    """The distance from the centre to the line through the apexes at the ends of each edge: the line's part
    paired with w, its moment about the centre, over its part wedged with w, its direction."""
    lines = apexes[..., edges[:, 0]] & apexes[..., edges[:, 1]]                # [..., edges] Bivector-
    return ((lines | core.mv.w).squared() / (lines ^ core.mv.w).squared()).square_root()


def test_circles_touch_along_every_edge_and_both_polyhedra_touch_the_sphere_under_every_boost():
    frames, faces, edges = scenarios.boosted()
    assert np.abs((frames | frames).to_array() - 1).max() < 1e-10
    assert np.abs((frames[:, edges[:, 0]] | frames[:, edges[:, 1]]).to_array() + 1).max() < 1e-10
    assert np.abs(centre_distances(core.apexes(frames), edges).to_array() - 1).max() < 1e-10
    duals = core.apexes(core.face_circles(frames, faces))                      # [frames, faces] Point-
    assert np.abs(centre_distances(duals, core.Mesh(frames.shape[-1], faces).edge_faces).to_array() - 1).max() < 1e-10


def test_the_centred_circles_balance_and_the_figure_draws():
    circles, faces, edges = scenarios.centred()
    tangents = core.touching(circles, edges)                                   # [edges] Vector+
    assert ((tangents / -(tangents | core.mv.w)).mean() - core.mv.w).norm() < 1e-4
    assert len(list(render.animate(circles[None], faces, edges))) == 1
