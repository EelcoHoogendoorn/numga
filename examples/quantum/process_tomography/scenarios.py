"""Four probes identify a rotation, phase noise, and relaxation, then predict repeated use."""

from __future__ import annotations

import numpy as np

from numga import stack
from examples.quantum.process_tomography import core

LABELS = ("Rotation", "Rotation + dephasing", "Rotation + relaxation")
TURNS = np.array([0.075, 0.075, 0.075])
PHASE_FLIP = np.array([0.0, 0.06, 0.0])
LOSS = np.array([0.0, 0.0, 0.10])
rotation_plane = (core.mv.yz + core.mv.zx + core.mv.xy) / np.sqrt(3)  # [] Bivector
directions = core.mv.vector([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]) / np.sqrt(3)  # [preparations] Vector
prepared = (core.one + directions) / 2               # [preparations] State
effects = prepared / 2                              # [outcomes] State
validation_directions = core.mv.vector([[1, 0, 0], [0, 1, 0], [0, 0, 1],
                                      [-1, 0, 0], [0, -1, 0], [0, 0, -1]])  # [heldout] Vector
BLOCH_LENGTHS = np.array([1.0, 0.7, 0.3, 1.0, 0.7, 0.0])
LATITUDES = 12
LONGITUDES = 2 * LATITUDES
STEPS = 48
FRAME_DURATION = 80
DESIGNS = ("Four tetrahedral probes", "Two opposite probes")


# --- math -----------------------------------------------------------------------------
def tomography() -> tuple[core.State, core.Scalar, core.Channel]:
    """The prepared states, their ideal measurement probabilities, and the reconstructed maps."""
    device = core.noisy_gate(TURNS, PHASE_FLIP, LOSS, rotation_plane)  # [channels] State <- State
    measured = core.probabilities(device, prepared, effects)  # [channels, preparations, outcomes] Scalar
    learned = core.reconstruct(prepared, effects, measured)  # [channels] State <- State
    return prepared, measured, learned


def validation(learned: core.Channel) -> tuple[core.Scalar, core.Scalar]:
    """Exact and predicted outcome probabilities for pure and mixed states absent from training."""
    heldout = (core.one + validation_directions * BLOCH_LENGTHS) / 2  # [heldout] State
    actual = core.noisy_gate(TURNS, PHASE_FLIP, LOSS, rotation_plane)  # [channels] State <- State
    observed = core.probabilities(actual, heldout, effects)  # [channels, heldout, outcomes] Scalar
    predicted = core.probabilities(learned, heldout, effects)  # [channels, heldout, outcomes] Scalar
    return observed, predicted


def completeness() -> core.Scalar:
    """How many independent state directions the tetrahedron and a single diameter determine."""
    full = (prepared * prepared.scalar_product(core.State)).sum(axis=-1)  # [] State <- State
    opposite = stack(((core.one + core.mv.z) / 2, (core.one - core.mv.z) / 2))  # [preparations] State
    incomplete = (opposite * opposite.scalar_product(core.State)).sum(axis=-1)  # [] (1 + z) <- State
    # Embed the missing output directions before taking a spectrum on the whole state space.
    return stack((full.svdvals(), incomplete.cast(core.Channel).svdvals()))  # [designs, directions] Scalar


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.quantum.process_tomography import render

    prepared, measured, learned = tomography()
    surface = core.sphere(LATITUDES, LONGITUDES)       # [latitudes + 1, longitudes + 1] State
    transformed = learned[:, None](prepared)          # [channels, preparations] State
    outputs = learned[:, None, None](surface)         # [channels, latitudes + 1, longitudes + 1] State

    save_figure(render.draw_experiment(prepared, measured, LABELS), "process_tomography_measurements")
    save_figure(render.draw_channels(surface, outputs, prepared, transformed, LABELS), "process_tomography_channels")
    save_figure(render.draw_predictions(*validation(learned), LABELS), "process_tomography_predictions")

    for label, spectrum in zip(DESIGNS, completeness().to_array()):
        print(f"{label}: {np.array2string(spectrum, precision=4, suppress_small=True)}")

    frames = ((combined[:, None, None](surface), combined[:, None](prepared))
              for combined in core.powers(learned, STEPS))
    save_animation(render.animate(surface, prepared, frames, LABELS), "process_tomography", FRAME_DURATION)
    # Gallery picture: this makes plots/tomography.gif for docs/extensors.md, tuned as a visual for the gallery. Don't change it; to experiment, copy it.
    # Every second use, each frame showing twice as long.
    frames = ((combined[:, None, None](surface), combined[:, None](prepared))
              for combined in core.powers(learned(learned), STEPS // 2))
    save_animation(render.animate_panels(surface, prepared, frames, LABELS, render.GALLERY_PANEL), "tomography",
                   2 * FRAME_DURATION)


if __name__ == "__main__":
    main()
