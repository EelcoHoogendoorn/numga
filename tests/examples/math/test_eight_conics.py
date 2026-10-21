"""Double contact, face concurrence, cube closure, and mirror covariance."""

import numpy as np

from examples.math.eight_conics import core, scenarios

ROUND_OFF = 1e-12


def test_deforming_cube_preserves_double_contact_and_face_concurrence():
    conics, chords, contacts, edges = scenarios.cube()
    first_tangents = conics[..., edges[:, 0]][..., None](contacts)
    second_tangents = conics[..., edges[:, 1]][..., None](contacts)
    face_edges = np.array([[0, 1, 3, 4], [0, 2, 5, 6], [1, 2, 7, 8],
                           [3, 5, 9, 10], [4, 7, 9, 11], [6, 8, 10, 11]])
    face_chords = chords[..., face_edges]
    face_centers = face_chords[..., 0] ^ face_chords[..., 1]

    # checks
    assert np.all(np.isfinite(contacts.kernel))
    assert np.all(np.linalg.norm((contacts[..., 0] & contacts[..., 1]).kernel, axis=-1) > 0.1)
    assert np.all(np.linalg.norm(first_tangents.kernel, axis=-1) > 0.1)
    assert np.all(np.abs((core.mv.w & face_centers).kernel) > 0.01)
    np.testing.assert_allclose((first_tangents & contacts).kernel, 0, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose((second_tangents & contacts).kernel, 0, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose((chords[..., None] & contacts).kernel, 0, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose((first_tangents ^ second_tangents).kernel, 0, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose((face_chords[..., 2:] ^ face_centers[..., None]).kernel,
                               0, atol=ROUND_OFF, rtol=0)


def test_three_completions_agree_and_reflect_with_their_inputs():
    scene_conics, scene_chords, contacts, edges = scenarios.cube()
    base = scene_conics[0, 0]
    initial_chords = scene_chords[0, :3]
    couplings = np.array([[-0.3, 0.2, 0.15], [0.4, -0.2, 0.1], [0.2, 0.45, -0.1]])
    wall = (core.mv.x + 0.7 * core.mv.y - 0.3 * core.mv.w).normalized()
    conics, chords, completions = core.configuration(base, initial_chords, couplings)
    mirror_base = wall >> base(wall << core.Point)
    mirror_initial_chords = wall >> initial_chords
    mirror_conics, mirror_chords, mirror_completions = core.configuration(
        mirror_base, mirror_initial_chords, couplings)

    # checks
    np.testing.assert_allclose((completions - completions[..., :1]).kernel,
                               0, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose((mirror_completions - mirror_completions[..., :1]).kernel,
                               0, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose((mirror_conics - (wall >> conics(wall << core.Point))).kernel,
                               0, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose((mirror_chords - (wall >> chords)).kernel,
                               0, atol=ROUND_OFF, rtol=0)
