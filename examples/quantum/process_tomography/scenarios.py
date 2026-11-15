"""Four tetrahedral probes identify rotation, dephasing, and relaxation channels and predict repeated gate passes."""

from __future__ import annotations

import numpy as np

from examples.quantum.process_tomography import core

mv = core.mv
labels = ("Rotation", "Rotation + dephasing", "Rotation + relaxation")
angles = np.array([0.075, 0.075, 0.075])
phase_flip = np.array([0.0, 0.06, 0.0])
loss = np.array([0.0, 0.0, 0.10])
rotation_plane = (mv.yz + mv.zx + mv.xy) / np.sqrt(3)                       # [] Bivector+
# Four calibration directions arranged as a regular tetrahedron, with analyzer planes facing them through the centre.
tetrahedron = mv.direction([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]) / np.sqrt(3)       # [4] Point-
analyzers = mv.vector([[1, 1, 1, 0], [1, -1, -1, 0], [-1, 1, -1, 0], [-1, -1, 1, 0]]) / np.sqrt(3)  # [4] Effect+
preparations = core.centre + tetrahedron                                     # [4] Point-
effects = (mv.w + analyzers) / 4                                             # [4] Effect+
validation_directions = mv.direction(
    [[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]],
)                                                                            # [6] Point-
bloch_lengths = np.array([1.0, 0.7, 0.3, 1.0, 0.7, 0.0])
n_latitudes = 12
n_longitudes = 2 * n_latitudes
n_uses = 48
frame_duration_ms = 40


# --- math -----------------------------------------------------------------------------
def tomography() -> tuple[core.Scalar, core.Channel]:
    """Tetrahedral probe measurements recover the three noisy gate channels from their ideal Born probabilities."""
    true_channels = core.noisy_gate(angles, phase_flip, loss, rotation_plane)  # [3] Point- <- Point-
    measurements = core.outcome_probabilities(true_channels, preparations, effects)  # [3, 4, 4] Scalar+
    learned_channels = core.reconstructed_channel(preparations, effects, measurements)  # [3] Point- <- Point-
    return measurements, learned_channels


def validation(learned_channels: core.Channel) -> tuple[core.Scalar, core.Scalar]:
    """Six axial pure and mixed states compare exact and reconstructed outcome probabilities."""
    validation_states = core.centre + validation_directions * bloch_lengths  # [6] Point-
    true_channels = core.noisy_gate(angles, phase_flip, loss, rotation_plane)  # [3] Point- <- Point-
    observed_probabilities = core.outcome_probabilities(true_channels, validation_states, effects)  # [3, 6, 4] Scalar+
    predicted_probabilities = core.outcome_probabilities(learned_channels, validation_states, effects)  # [3, 6, 4] Scalar+
    return observed_probabilities, predicted_probabilities


def completeness() -> tuple[core.Antiscalar, core.Line]:
    """Four tetrahedral probes join to a volume antiscalar while two opposite probes join only to an axial line."""
    south = mv.point([0, 0, -1])                                             # [] Point-
    tetrahedral_span = preparations[0] & preparations[1] & preparations[2] & preparations[3]  # [] Antiscalar-
    opposite_span = core.north & south                                       # [] Line-
    return tetrahedral_span, opposite_span


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.quantum.process_tomography import render
    from examples.util.animation import save_animation, save_figure

    measurements, learned_channels = tomography()
    surface = core.pure_sphere(n_latitudes, n_longitudes)                    # [n_latitudes + 1, n_longitudes] Point-
    probe_images = learned_channels[:, None](preparations)                   # [3, 4] Point-
    outputs = learned_channels[:, None, None](surface)                       # [3, n_latitudes + 1, n_longitudes] Point-
    observed_probabilities, predicted_probabilities = validation(learned_channels)
    reading = render.ball_positions

    save_figure(render.experiment_figure(preparations, measurements, labels, reading), "process_tomography_measurements")
    save_figure(
        render.channels_figure(surface, outputs, preparations, probe_images, labels, reading),
        "process_tomography_channels",
    )
    save_figure(
        render.predictions_figure(observed_probabilities, predicted_probabilities, labels),
        "process_tomography_predictions",
    )

    tetrahedral_span, opposite_span = completeness()
    print("Four tetrahedral probes join to a volume:\n", tetrahedral_span.coefficient_table())
    print("Two opposite probes join to a line:\n", opposite_span.coefficient_table())

    frames = (
        (channel_power[:, None, None](surface), channel_power[:, None](preparations))
        for channel_power in core.channel_powers(learned_channels, n_uses)
    )
    save_animation(
        render.channel_animation(surface, preparations, labels, reading, render.panel_size, frames),
        "process_tomography", frame_duration_ms,
    )

    # checks
    assert (measurements >= -1e-12).all()
    assert ((measurements.sum(axis=-1) - 1).abs() < 1e-12).all()
    assert ((core.outcome_probabilities(learned_channels, preparations, effects) - measurements).abs() < 1e-12).all()
    assert ((observed_probabilities - predicted_probabilities).abs() < 1e-12).all()
    assert (((mv.w & outputs) - 1).abs() < 1e-12).all()
    assert (((outputs[0] - core.centre).antinorm() - 1).abs() < 1e-12).all()
    assert ((outputs[1:] - core.centre).antinorm() <= 1 + 1e-12).all()
    assert tetrahedral_span.norm() > 1.0
    assert (opposite_span & core.centre).norm() < 1e-12
    assert ((opposite_span & preparations).norm() > 0.1).all()


def pinned() -> None:
    """Repeated applications of the reconstructed channels write the gallery animation."""
    from examples.quantum.process_tomography import render
    from examples.util.animation import pin_animation

    _, learned_channels = tomography()
    surface = core.pure_sphere(n_latitudes, n_longitudes)                    # [n_latitudes + 1, n_longitudes] Point-
    frames = (
        (channel_power[:, None, None](surface), channel_power[:, None](preparations))
        for channel_power in core.channel_powers(learned_channels, n_uses)
    )
    pin_animation(
        render.channel_animation(surface, preparations, labels, render.ball_positions, render.gallery_panel_size, frames),
        "tomography", frame_duration_ms,
    )


if __name__ == "__main__":
    main()
