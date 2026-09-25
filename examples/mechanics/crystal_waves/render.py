"""Phonon-focusing images and wave-surface sections: vectors read out for matplotlib."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.mechanics.crystal_waves import core
from examples.mechanics.crystal_waves.scenarios import MODES

MODE_COLOURS = ("#c0392b", "#2e86c1", "#7d3c98")


def components(vectors: core.Vector) -> np.ndarray:
    """Euclidean components of vectors."""
    return vectors.cast(core.ga.subspace("x y z")).kernel


def draw_focusing(velocities: list[core.Vector], names: list[str], bins: int) -> plt.Figure:
    """One row per crystal, one column per wave: how densely the energy of evenly spread headings
    lands on a cube face, seen from a source at the centre of the opposite face."""
    figure, grid = plt.subplots(len(names), len(MODES), figsize=(4 * len(MODES), 4 * len(names)))
    for row, velocity, name in zip(np.atleast_2d(grid), velocities, names):
        flow = components(velocity)                                    # [count, waves, 3]
        for ax, wave, mode in zip(row, np.moveaxis(flow, 1, 0), MODES):
            ahead = wave[:, 2] > 0
            face = wave[ahead, :2] / wave[ahead, 2:]                   # where each ray meets the face z = 1
            density, _, _ = np.histogram2d(face[:, 0], face[:, 1], bins=bins, range=[[-1.5, 1.5], [-1.5, 1.5]])
            ax.imshow(np.log1p(density.T), cmap="inferno", origin="lower", extent=[-1.5, 1.5, -1.5, 1.5])
            ax.set_title(f"{name}: {mode}", fontsize=10)
            ax.set_axis_off()
    figure.tight_layout()
    return figure


def draw_wave_fronts(velocities: list[core.Vector], names: list[str]) -> plt.Figure:
    """One panel per crystal: where each wave's energy is after unit time, in the cube face."""
    figure, row = plt.subplots(1, len(names), figsize=(5 * len(names), 5))
    for ax, velocity, name in zip(np.atleast_1d(row), velocities, names):
        flow = components(velocity)                                    # [count, waves, 3]
        for wave, mode, colour in zip(np.moveaxis(flow, 1, 0), MODES, MODE_COLOURS):
            ax.plot(wave[:, 0], wave[:, 1], ".", markersize=1.0, color=colour, label=mode)
        ax.set_aspect("equal")
        ax.set_title(name)
        ax.set_xlabel("km/s along [100]")
        ax.legend(markerscale=8, loc="upper right", fontsize=8)
    figure.tight_layout()
    return figure
