"""Coverage tests for double contact, face concurrence, cube closure, mirror covariance, and rendering."""

import matplotlib.pyplot as plt
import numpy as np

from numga import stack
from examples.math.eight_conics import core, render, scenarios

tolerance = 1e-9
n_test_frames = 2


def test_deforming_cube_preserves_double_contact_and_face_concurrence():
    states = list(scenarios.cube_states(n_test_frames))
    quadrics = stack([state[0] for state in states])                        # [n_test_frames, 8] Plane+ <- Point-
    contact_planes = stack([state[1] for state in states])                  # [n_test_frames, 12] Plane+
    contacts = stack([state[2] for state in states])                        # [n_test_frames, 12, 2] Point-

    first_tangents = quadrics[:, core.edges[:, 0]][..., None](contacts)     # [n_test_frames, 12, 2] Plane+
    second_tangents = quadrics[:, core.edges[:, 1]][..., None](contacts)    # [n_test_frames, 12, 2] Plane+
    face_edges = np.array([
        [0, 1, 3, 4], [0, 2, 5, 6], [1, 2, 7, 8],
        [3, 5, 9, 10], [4, 7, 9, 11], [6, 8, 10, 11],
    ])
    face_planes = contact_planes[:, face_edges]                             # [n_test_frames, 6, 4] Plane+
    face_meets = face_planes[..., 0] ^ face_planes[..., 1]                  # [n_test_frames, 6] Bivector+
    basis_planes = stack((core.mv.x, core.mv.y, core.mv.w))                 # [3] Plane+
    antiscalar = core.mv.xyw.dual()                                         # [] Scalar-

    # checks
    assert (basis_planes & contacts[..., None]).isfinite().all()
    assert ((contacts[..., 1] - contacts[..., 0]).antinorm() > 0.1).all()
    assert (first_tangents.norm() > 0.1).all()
    assert (((face_meets ^ core.mv.w) & antiscalar).abs() > 0.01).all()
    assert ((first_tangents & contacts).abs() < tolerance).all()
    assert ((second_tangents & contacts).abs() < tolerance).all()
    assert ((contact_planes[..., None] & contacts).abs() < tolerance).all()
    tangent_wedge = ((first_tangents ^ second_tangents)[..., None] ^ basis_planes) & antiscalar  # [n_test_frames, 12, 2, 3] Scalar+
    assert (tangent_wedge.abs() < tolerance).all()
    face_concurrence = (face_planes[..., 2:] ^ face_meets[..., None]) & antiscalar  # [n_test_frames, 6, 2] Scalar+
    assert (face_concurrence.abs() < tolerance).all()


def test_three_completions_agree_and_reflect_with_their_inputs():
    first_state = next(scenarios.cube_states(1))
    base = first_state[0][0]                                                # [] Plane+ <- Point-
    initial_planes = first_state[1][:3]                                     # [3] Plane+
    couplings = np.array([[-0.3, 0.2, 0.15], [0.4, -0.2, 0.1], [0.2, 0.45, -0.1]])
    wall = (core.mv.x + 0.7 * core.mv.y - 0.3 * core.mv.w).normalized()     # [] Plane+
    basis_points = stack((core.mv.x.dual(), core.mv.y.dual(), core.mv.w.dual()))  # [3] Point-

    quadrics, contact_planes, completions = core.configuration(base, initial_planes, couplings)
    mirror_base = wall >> base(wall << core.Point)                          # [] Plane+ <- Point-
    mirror_initial_planes = wall >> initial_planes                          # [3] Plane+
    mirror_quadrics, mirror_planes, mirror_completions = core.configuration(
        mirror_base, mirror_initial_planes, couplings,
    )

    # checks
    completion_residual = (
        (completions - completions[..., :1])[..., None](basis_points)[..., None] & basis_points
    )                                                                       # [3, 3, 3, 3] Scalar+
    mirror_completion_residual = (
        (mirror_completions - mirror_completions[..., :1])[..., None](basis_points)[..., None] & basis_points
    )                                                                       # [3, 3, 3, 3] Scalar+
    reflected_quadrics = wall >> quadrics(wall << core.Point)               # [3, 8] Plane+ <- Point-
    quadric_mirror_residual = (
        (mirror_quadrics - reflected_quadrics)[..., None](basis_points)[..., None] & basis_points
    )                                                                       # [3, 8, 3, 3] Scalar+
    plane_mirror_residual = (mirror_planes - (wall >> contact_planes))[..., None] & basis_points  # [3, 12, 3] Scalar+

    assert (completion_residual.abs() < tolerance).all()
    assert (mirror_completion_residual.abs() < tolerance).all()
    assert (quadric_mirror_residual.abs() < tolerance).all()
    assert (plane_mirror_residual.abs() < tolerance).all()


def test_rendering_produces_double_contact_figure_and_animation_frame():
    sphere, contact_quadric, pair_plane, pair_contacts = scenarios.double_contact_pair()
    figure = render.double_contact_figure(sphere, contact_quadric, pair_contacts)
    frames = list(render.eighth_conic_animation(core.edges, scenarios.closing_edges, scenarios.cube_states(1)))

    # checks
    assert len(figure.axes) == 1
    assert len(frames) == 1
    assert frames[0].ndim == 3 and frames[0].shape[-1] == 3
    plt.close(figure)
