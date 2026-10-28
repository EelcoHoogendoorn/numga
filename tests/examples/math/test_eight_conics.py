"""Double contact, face concurrence, cube closure, and mirror covariance."""

import numpy as np

from examples.math.eight_conics import core, scenarios

ROUND_OFF = 1e-12


def test_deforming_cube_preserves_double_contact_and_face_concurrence():
    quadrics, contact_planes, contacts, edges = scenarios.cube()
    first_tangents = quadrics[..., edges[:, 0]][..., None](contacts)
    second_tangents = quadrics[..., edges[:, 1]][..., None](contacts)
    face_edges = np.array([[0, 1, 3, 4], [0, 2, 5, 6], [1, 2, 7, 8],
                           [3, 5, 9, 10], [4, 7, 9, 11], [6, 8, 10, 11]])
    face_planes = contact_planes[..., face_edges]
    face_centers = face_planes[..., 0] ^ face_planes[..., 1]

    # checks
    assert np.all(np.isfinite(contacts.kernel))
    assert np.all(np.linalg.norm((contacts[..., 0] & contacts[..., 1]).kernel, axis=-1) > 0.1)
    assert np.all(np.linalg.norm(first_tangents.kernel, axis=-1) > 0.1)
    assert np.all(np.abs((core.mv.w & face_centers).kernel) > 0.01)
    np.testing.assert_allclose((first_tangents & contacts).kernel, 0, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose((second_tangents & contacts).kernel, 0, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose((contact_planes[..., None] & contacts).kernel, 0, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose((first_tangents ^ second_tangents).kernel, 0, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose((face_planes[..., 2:] ^ face_centers[..., None]).kernel,
                               0, atol=ROUND_OFF, rtol=0)


def test_three_completions_agree_and_reflect_with_their_inputs():
    scene_quadrics, scene_planes, contacts, edges = scenarios.cube()
    base = scene_quadrics[0, 0]
    initial_planes = scene_planes[0, :3]
    couplings = np.array([[-0.3, 0.2, 0.15], [0.4, -0.2, 0.1], [0.2, 0.45, -0.1]])
    wall = (core.mv.x + 0.7 * core.mv.y - 0.3 * core.mv.w).normalized()
    quadrics, contact_planes, completions = core.configuration(base, initial_planes, couplings)
    mirror_base = wall >> base(wall << core.Point)
    mirror_initial_planes = wall >> initial_planes
    mirror_quadrics, mirror_planes, mirror_completions = core.configuration(
        mirror_base, mirror_initial_planes, couplings)

    # checks
    np.testing.assert_allclose((completions - completions[..., :1]).kernel,
                               0, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose((mirror_completions - mirror_completions[..., :1]).kernel,
                               0, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose((mirror_quadrics - (wall >> quadrics(wall << core.Point))).kernel,
                               0, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose((mirror_planes - (wall >> contact_planes)).kernel,
                               0, atol=ROUND_OFF, rtol=0)
