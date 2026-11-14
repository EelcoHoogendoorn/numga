"""Tests physical channel action, independent-frame tomography, repeated composition, and figure rendering."""

import matplotlib.pyplot as plt
import numpy as np

from examples.quantum.process_tomography import core, render, scenarios


def analytical_bloch(
    bloch: np.ndarray, angles: np.ndarray, phase_flip: np.ndarray, loss: np.ndarray, axis: np.ndarray,
) -> np.ndarray:
    """Dephasing, amplitude damping, and Rodrigues' rotation give the exact outgoing Bloch vectors."""
    transverse = (1 - 2 * phase_flip) * np.sqrt(1 - loss)
    contracted_vectors = np.stack((
        transverse[:, None] * bloch[..., 0],
        transverse[:, None] * bloch[..., 1],
        (1 - loss[:, None]) * bloch[..., 2] + loss[:, None],
    ), axis=-1)
    cosine, sine = np.cos(angles[:, None, None]), np.sin(angles[:, None, None])
    return (
        cosine * contracted_vectors
        + sine * np.cross(axis, contracted_vectors)
        + (1 - cosine) * axis * np.sum(axis * contracted_vectors, axis=-1, keepdims=True)
    )


def test_channels_match_independent_bloch_law_and_normalized_born_probabilities():
    angles = np.array([0.0, 0.47, -0.9, 0.3])
    phase_flip = np.array([0.0, 0.22, 0.08, 0.5])
    loss = np.array([0.0, 0.45, 1.0, 0.3])
    axis = np.array([2.0, -1.0, 3.0]) / np.sqrt(14)
    rotation_plane = (2 * core.mv.yz - core.mv.zx + 3 * core.mv.xy) / np.sqrt(14)
    channels = core.noisy_gate(angles, phase_flip, loss, rotation_plane)

    rng = np.random.default_rng(180)
    raw_directions = rng.normal(size=(12, 3))
    raw_directions /= np.linalg.norm(raw_directions, axis=-1, keepdims=True)
    bloch = raw_directions * np.linspace(0, 1, len(raw_directions))[:, None]
    preparations = core.centre + core.mv.direction(bloch)
    expected = analytical_bloch(np.broadcast_to(bloch, (len(angles),) + bloch.shape), angles, phase_flip, loss, axis)
    outputs = channels[:, None](preparations)
    expected_states = core.centre + core.mv.direction(expected)

    assert ((outputs - expected_states).antinorm() < 1e-8).all()
    assert (((core.mv.w & outputs) - 1).abs() < 1e-8).all()
    assert ((outputs - core.centre).antinorm() <= 1 + 1e-8).all()

    detector_directions = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]) / np.sqrt(3)
    analyzers = core.mv.vector(np.concatenate([detector_directions, np.zeros((4, 1))], axis=-1))
    effects = (core.mv.w + analyzers) / 4
    probabilities = core.outcome_probabilities(channels, preparations, effects)
    expected_probabilities = (1 + np.einsum("cni,mi->cnm", expected, detector_directions)) / 4
    assert ((probabilities - expected_probabilities).abs() < 1e-8).all()
    assert ((probabilities.sum(axis=-1) - 1).abs() < 1e-8).all()
    assert (probabilities >= -1e-8).all()


def test_reconstruction_predicts_mixed_states_with_independent_measurement_frame():
    angles = np.array([0.31, -0.76])
    phase_flip = np.array([0.12, 0.07])
    loss = np.array([0.26, 0.54])
    rotation_plane = (core.mv.yz + core.mv.xy) / np.sqrt(2)
    true_channels = core.noisy_gate(angles, phase_flip, loss, rotation_plane)

    # Four mixed preparations in general position paired with an independently rotated detector frame.
    probe_offsets = core.mv.direction([[0.9, 0.1, 0.0], [-0.2, 0.8, 0.1], [0.1, -0.3, -0.7], [-0.5, -0.4, 0.6]])
    preparations = core.centre + probe_offsets
    detector_rotor = (core.mv.xy * -0.23).exp() * (core.mv.yz * 0.17).exp()
    effects = detector_rotor >> scenarios.effects
    measurements = core.outcome_probabilities(true_channels, preparations, effects)
    learned_channels = core.reconstructed_channel(preparations, effects, measurements)

    rng = np.random.default_rng(972)
    raw_directions = rng.normal(size=(12, 3))
    raw_directions /= np.linalg.norm(raw_directions, axis=-1, keepdims=True)
    bloch = raw_directions * np.linspace(0, 1, len(raw_directions))[:, None]
    validation_states = core.centre + core.mv.direction(bloch)
    predicted_states = learned_channels[:, None](validation_states)
    exact_states = true_channels[:, None](validation_states)
    assert ((predicted_states - exact_states).antinorm() < 1e-12).all()
    assert (((core.mv.w & predicted_states) - 1).abs() < 1e-12).all()


def test_learned_channels_predict_unseen_measurements_and_repeated_applications():
    measurements, learned_channels = scenarios.tomography()
    observed_probabilities, predicted_probabilities = scenarios.validation(learned_channels)
    assert ((observed_probabilities - predicted_probabilities).abs() < 1e-12).all()
    assert (
        (core.outcome_probabilities(learned_channels, scenarios.preparations, scenarios.effects) - measurements).abs()
        < 1e-12
    ).all()

    bloch = np.array([[0.2, -0.3, 0.4], [-0.6, 0.1, 0.2], [0.0, 0.0, -1.0]])
    states = core.centre + core.mv.direction(bloch)
    expected = np.broadcast_to(bloch, (len(scenarios.angles),) + bloch.shape)
    axis = np.ones(3) / np.sqrt(3)
    for channel_power in core.channel_powers(learned_channels, 4):
        expected = analytical_bloch(expected, scenarios.angles, scenarios.phase_flip, scenarios.loss, axis)
        expected_states = core.centre + core.mv.direction(expected)
        mapped_states = channel_power[:, None](states)
        assert ((mapped_states - expected_states).antinorm() < 1e-8).all()
        assert (((core.mv.w & mapped_states) - 1).abs() < 1e-8).all()


def test_completeness_distinguishes_spanning_tetrahedron_from_axial_probes():
    tetrahedral_span, opposite_span = scenarios.completeness()
    assert (tetrahedral_span.norm() - 16 / (3 * np.sqrt(3))).abs() < 1e-12
    assert (opposite_span & core.centre).norm() < 1e-12
    assert ((opposite_span & scenarios.preparations).norm() > 0.1).all()


def test_figures_and_single_animation_frame_render():
    measurements, learned_channels = scenarios.tomography()
    preparations, channel_labels, reading = scenarios.preparations, scenarios.labels, render.ball_positions
    surface = core.pure_sphere(4, 6)
    fig_exp = render.experiment_figure(preparations, measurements, channel_labels, reading)
    fig_ch = render.channels_figure(
        surface, learned_channels[:, None, None](surface), preparations, learned_channels[:, None](preparations),
        channel_labels, reading,
    )
    fig_pred = render.predictions_figure(*scenarios.validation(learned_channels), channel_labels)
    frames = (
        (channel_power[:, None, None](surface), channel_power[:, None](preparations))
        for channel_power in core.channel_powers(learned_channels, 1)
    )
    frame = next(render.channel_animation(surface, preparations, channel_labels, reading, render.panel_size, frames))
    assert frame.ndim == 3 and frame.shape[2] == 3
    plt.close(fig_exp)
    plt.close(fig_ch)
    plt.close(fig_pred)
