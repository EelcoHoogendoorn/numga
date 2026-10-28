"""Four probes identify a rotation, phase noise, and relaxation, then predict repeated use."""

from __future__ import annotations

import numpy as np

from examples.quantum.process_tomography import core

mv = core.mv
LABELS = ("Rotation", "Rotation + dephasing", "Rotation + relaxation")
TURNS = np.array([0.075, 0.075, 0.075])
PHASE_FLIP = np.array([0.0, 0.06, 0.0])
LOSS = np.array([0.0, 0.0, 0.10])
rotation_plane = (mv.yz + mv.zx + mv.xy) / np.sqrt(3)                       # [] Bivector+
# Four calibration directions, arranged as a tetrahedron, and the detector's planes through the centre facing them.
tetrahedron = mv.direction([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]) / np.sqrt(3)       # [preparations] Point-
analyzers = mv.vector([[1, 1, 1, 0], [1, -1, -1, 0], [-1, 1, -1, 0], [-1, -1, 1, 0]]) / np.sqrt(3)  # [outcomes] Effect+
prepared = core.centre + tetrahedron                                         # [preparations] Point-
# The plane at infinity reads a state's weight; with the analyzers' planes, four effects that add up to it.
effects = (mv.w + analyzers) / 4                                             # [outcomes] Effect+
validation_directions = mv.direction([[1, 0, 0], [0, 1, 0], [0, 0, 1],
                                      [-1, 0, 0], [0, -1, 0], [0, 0, -1]])  # [heldout] Point-
BLOCH_LENGTHS = np.array([1.0, 0.7, 0.3, 1.0, 0.7, 0.0])
N_LATITUDES = 12
N_LONGITUDES = 2 * N_LATITUDES
N_USES = 48
FRAME_DURATION = 40


# --- math -----------------------------------------------------------------------------
def tomography() -> tuple[core.Scalar, core.Channel]:
    """The ideal measurement probabilities of the prepared states, and the reconstructed maps."""
    device = core.noisy_gate(TURNS, PHASE_FLIP, LOSS, rotation_plane)        # [channels] Point- <- Point-
    measured = core.probabilities(device, prepared, effects)                 # [channels, preparations, outcomes] Scalar+
    learned = core.reconstruct(prepared, effects, measured)                  # [channels] Point- <- Point-
    return measured, learned


def validation(learned: core.Channel) -> tuple[core.Scalar, core.Scalar]:
    """Exact and predicted outcome probabilities for pure and mixed states absent from training."""
    heldout = core.centre + validation_directions * BLOCH_LENGTHS             # [heldout] Point-
    actual = core.noisy_gate(TURNS, PHASE_FLIP, LOSS, rotation_plane)        # [channels] Point- <- Point-
    observed = core.probabilities(actual, heldout, effects)                  # [channels, heldout, outcomes] Scalar+
    predicted = core.probabilities(learned, heldout, effects)                # [channels, heldout, outcomes] Scalar+
    return observed, predicted


def completeness() -> tuple[core.Scalar, core.Line]:
    """What the tetrahedron and a single diameter of probes join to: a volume, which every state lies in, and the
    line of the field, which only the states on it lie in."""
    south = mv.point([0, 0, -1])                                             # [] Point-
    tetrahedral_span = prepared[0] & prepared[1] & prepared[2] & prepared[3]  # [] Scalar-
    opposite_span = core.north & south                                       # [] Line-
    return tetrahedral_span, opposite_span


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.quantum.process_tomography import render

    measured, learned = tomography()
    surface = core.sphere(N_LATITUDES, N_LONGITUDES)                         # [latitudes + 1, longitudes + 1] Point-
    transformed = learned[:, None](prepared)                                 # [channels, preparations] Point-
    outputs = learned[:, None, None](surface)                                # [channels, latitudes + 1, longitudes + 1] Point-
    reading = render.ball_positions

    save_figure(render.draw_experiment(prepared, measured, LABELS, reading), "process_tomography_measurements")
    save_figure(render.draw_channels(surface, outputs, prepared, transformed, LABELS, reading),
                "process_tomography_channels")
    save_figure(render.draw_predictions(*validation(learned), LABELS), "process_tomography_predictions")

    tetrahedral_span, opposite_span = completeness()
    print("Four tetrahedral probes join to a volume:\n", tetrahedral_span.coefficient_table())
    print("Two opposite probes join to a line:\n", opposite_span.coefficient_table())

    frames = ((combined[:, None, None](surface), combined[:, None](prepared))
              for combined in core.powers(learned, N_USES))
    save_animation(render.animate(surface, prepared, frames, LABELS, reading, render.PANEL), "process_tomography",
                   FRAME_DURATION)
    # Gallery picture: this makes plots/tomography.gif for docs/extensors.md, tuned as a visual for the gallery. Don't change it; to experiment, copy it.
    frames = ((combined[:, None, None](surface), combined[:, None](prepared))
              for combined in core.powers(learned, N_USES))
    save_animation(render.animate(surface, prepared, frames, LABELS, reading, render.GALLERY_PANEL), "tomography",
                   FRAME_DURATION)


if __name__ == "__main__":
    main()
