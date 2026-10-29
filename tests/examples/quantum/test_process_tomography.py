"""Physical channel action, independent tomography, and predictions under composition."""

import numpy as np

from numga import stack
from examples.quantum.process_tomography import core, scenarios


def analytical_bloch(bloch: np.ndarray, angles: np.ndarray, phase_flip: np.ndarray,
                     loss: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Dephasing and amplitude damping followed by Rodrigues' spatial rotation."""
    transverse = (1 - 2 * phase_flip) * np.sqrt(1 - loss)
    damped = np.stack((
        transverse[:, None] * bloch[..., 0],
        transverse[:, None] * bloch[..., 1],
        (1 - loss[:, None]) * bloch[..., 2] + loss[:, None],
    ), axis=-1)
    cosine, sine = np.cos(angles[:, None, None]), np.sin(angles[:, None, None])
    return (cosine * damped + sine * np.cross(axis, damped)
            + (1 - cosine) * axis * np.sum(axis * damped, axis=-1, keepdims=True))


def test_channels_match_the_independent_bloch_law_and_normalized_born_probabilities():
    angles = np.array([0.0, 0.47, -0.9, 0.3])
    phase_flip = np.array([0.0, 0.22, 0.08, 0.5])
    loss = np.array([0.0, 0.45, 1.0, 0.3])
    axis = np.array([2.0, -1.0, 3.0]) / np.sqrt(14)
    rotation_plane = (2 * core.mv.yz - core.mv.zx + 3 * core.mv.xy) / np.sqrt(14)
    device = core.noisy_gate(angles, phase_flip, loss, rotation_plane)
    rng = np.random.default_rng(180)
    directions = rng.normal(size=(29, 3))
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    bloch = directions * np.linspace(0, 1, len(directions))[:, None]
    prepared = core.centre + core.mv.direction(bloch)
    expected = analytical_bloch(np.broadcast_to(bloch, (len(angles),) + bloch.shape),
                                angles, phase_flip, loss, axis)
    outputs = device[:, None](prepared)
    expected_states = core.centre + core.mv.direction(expected)
    np.testing.assert_allclose((outputs - expected_states).kernel, 0, atol=1e-8)
    np.testing.assert_allclose((core.mv.w & outputs).to_array(), 1, atol=1e-8)

    actual_bloch = (stack((core.mv.x, core.mv.y, core.mv.z)) & outputs[..., None]).to_array()
    assert np.max(np.linalg.norm(actual_bloch, axis=-1)) <= 1 + 1e-8

    detector_directions = np.array([[1, 1, 1], [1, -1, -1],
                                    [-1, 1, -1], [-1, -1, 1]]) / np.sqrt(3)
    analyzers = core.mv.vector(np.concatenate([detector_directions, np.zeros((4, 1))], axis=-1))
    effects = (core.mv.w + analyzers) / 4
    probabilities = core.probabilities(device, prepared, effects).to_array()
    expected_probabilities = (1 + np.einsum("cni,mi->cnm", expected,
                                             detector_directions)) / 4
    np.testing.assert_allclose(probabilities, expected_probabilities, atol=1e-8)
    np.testing.assert_allclose(probabilities.sum(axis=-1), 1, atol=1e-8)
    assert probabilities.min() >= -1e-8


def test_reconstruction_predicts_mixed_states_with_an_independent_measurement_frame():
    angles = np.array([0.31, -0.76])
    phase_flip = np.array([0.12, 0.07])
    loss = np.array([0.26, 0.54])
    rotation_plane = (core.mv.yz + core.mv.xy) / np.sqrt(2)
    device = core.noisy_gate(angles, phase_flip, loss, rotation_plane)

    # Four mixed preparations in general position, while the four detector directions are rotated
    # independently. The reconstruction cannot rely on matching the two frames.
    directions = core.mv.direction([[0.9, 0.1, 0.0], [-0.2, 0.8, 0.1],
                                    [0.1, -0.3, -0.7], [-0.5, -0.4, 0.6]])
    prepared = core.centre + directions
    rotation = (core.mv.xy * -0.23).exp() * (core.mv.yz * 0.17).exp()
    effects = rotation >> scenarios.effects
    measured = core.probabilities(device, prepared, effects)
    learned = core.reconstruct(prepared, effects, measured)

    rng = np.random.default_rng(972)
    directions = rng.normal(size=(37, 3))
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    bloch = directions * np.linspace(0, 1, len(directions))[:, None]
    unseen = core.centre + core.mv.direction(bloch)
    np.testing.assert_allclose((learned[:, None](unseen) - device[:, None](unseen)).kernel,
                               0, atol=1e-12)


def test_learned_channels_predict_unseen_measurements_and_repeated_applications():
    measured, learned = scenarios.tomography()
    observed, predicted = scenarios.validation(learned)
    np.testing.assert_allclose((observed - predicted).kernel, 0, atol=1e-12)
    np.testing.assert_allclose(
        (core.probabilities(learned, scenarios.prepared, scenarios.effects) - measured).kernel,
        0, atol=1e-12,
    )

    bloch = np.array([[0.2, -0.3, 0.4], [-0.6, 0.1, 0.2], [0.0, 0.0, -1.0]])
    states = core.centre + core.mv.direction(bloch)
    expected = np.broadcast_to(bloch, (len(scenarios.TURNS),) + bloch.shape)
    axis = np.ones(3) / np.sqrt(3)
    n_uses = 4
    for accumulated in core.powers(learned, n_uses):
        expected = analytical_bloch(expected, scenarios.TURNS, scenarios.PHASE_FLIP,
                                    scenarios.LOSS, axis)
        expected_states = core.centre + core.mv.direction(expected)
        np.testing.assert_allclose((accumulated[:, None](states) - expected_states).kernel,
                                   0, atol=1e-8)


def test_completeness_distinguishes_a_spanning_tetrahedron_from_two_axial_probes():
    tetrahedral_span, opposite_span = scenarios.completeness()
    # Four probes join to six times the volume of their tetrahedron; two opposite ones only to the line through them,
    # which the completely mixed state lies on and the tetrahedron's states do not.
    np.testing.assert_allclose(np.abs(tetrahedral_span.to_array()), 16 / (3 * np.sqrt(3)), atol=1e-12)
    np.testing.assert_allclose((opposite_span & core.centre).kernel, 0, atol=1e-12)
    assert np.abs((opposite_span & scenarios.prepared).kernel).max() > 0.1


def test_figures_and_composed_channel_animation_draw_without_saving():
    import matplotlib.pyplot as plt

    from examples.quantum.process_tomography import render

    measured, learned = scenarios.tomography()
    observed, predicted = scenarios.validation(learned)
    prepared = scenarios.prepared
    surface = core.sphere(8, 16)
    # The display surface consists of pure states, of weight one on the unit sphere, before any channel acts.
    np.testing.assert_allclose((core.mv.w & surface).to_array(), 1, atol=1e-8)
    np.testing.assert_allclose((core.centre & render.ball_positions(surface)).norm().to_array(), 1, atol=1e-8)
    figures = (
        render.draw_experiment(prepared, measured, scenarios.LABELS, render.ball_positions),
        render.draw_predictions(observed, predicted, scenarios.LABELS),
    )
    for figure in figures:
        figure.canvas.draw()
        plt.close(figure)

    states = ((accumulated[:, None, None](surface), accumulated[:, None](prepared))
              for accumulated in core.powers(learned, 2))
    frames = list(render.animate(surface, prepared, states, scenarios.LABELS, render.ball_positions, render.PANEL))
    assert frames[0].ndim == 3 and frames[0].shape == frames[1].shape
    assert not np.array_equal(frames[0], frames[1])
