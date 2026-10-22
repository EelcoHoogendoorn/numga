"""Phonon-focusing images and wave-surface sections: vectors read out for matplotlib."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.mechanics.crystal_waves import core
from examples.mechanics.crystal_waves.scenarios import MODES

MODE_COLOURS = ("#c0392b", "#2e86c1", "#7d3c98")
CROSS_COLOUR, LABEL_COLOUR = "0.85", "0.4"
# The margin around the wave fronts, as a share of their reach.
MARGIN = 0.2


def components(vectors: core.Vector) -> np.ndarray:
    """Euclidean components of vectors."""
    return vectors.cast(core.ga.subspace("x y z")).kernel


def cross(ax: plt.Axes, limit: float, across: str, up: str) -> None:
    """A square plane of half width `limit` without scales: a faint cross through the origin, its lines named at
    their ends."""
    ax.axhline(0, color=CROSS_COLOUR, linewidth=0.7, zorder=0)
    ax.axvline(0, color=CROSS_COLOUR, linewidth=0.7, zorder=0)
    ax.text(limit, 0, across, ha="right", va="top", fontsize=9, color=LABEL_COLOUR)
    ax.text(0, limit, f" {up}", ha="left", va="top", fontsize=9, color=LABEL_COLOUR)
    ax.set(xlim=(-limit, limit), ylim=(-limit, limit), aspect="equal")
    ax.set_axis_off()


def draw_focusing(velocities: core.Vector, names: list[str], bins: int) -> plt.Figure:
    """One row per crystal, one column per wave: how densely the energy of evenly spread headings
    lands on a cube face, seen from a source at the centre of the opposite face."""
    figure, grid = plt.subplots(len(names), len(MODES), figsize=(4 * len(MODES), 4 * len(names)))
    for row, velocity, name in zip(np.atleast_2d(grid), velocities, names):
        flow = components(velocity)                                    # [count, waves, 3]
        for ax, wave, mode in zip(row, np.moveaxis(flow, 1, 0), MODES):
            ahead = wave[:, 2] > 0
            # Where each ray meets the face one unit along z.
            face = wave[ahead, :2] / wave[ahead, 2:]
            density, _, _ = np.histogram2d(face[:, 0], face[:, 1], bins=bins, range=[[-1.5, 1.5], [-1.5, 1.5]])
            ax.imshow(np.log1p(density.T), cmap="inferno", origin="lower", extent=[-1.5, 1.5, -1.5, 1.5])
            ax.set_title(f"{name}: {mode}", fontsize=10)
            ax.set_axis_off()
    figure.tight_layout()
    return figure


def draw_wave_fronts(velocities: core.Vector, names: list[str]) -> plt.Figure:
    """One panel per crystal: where each wave's energy is after unit time, in the cube face, in the directions of
    the crystal."""
    figure, row = plt.subplots(1, len(names), figsize=(5 * len(names), 5.4), layout="constrained")
    for ax, velocity, name in zip(np.atleast_1d(row), velocities, names):
        flow = components(velocity)                                    # [count, waves, 3]
        for wave, mode, colour in zip(np.moveaxis(flow, 1, 0), MODES, MODE_COLOURS):
            ax.plot(wave[:, 0], wave[:, 1], ".", markersize=1.0, color=colour, label=mode)
        cross(ax, (1 + MARGIN) * np.abs(flow[..., :2]).max(), "[100]", "[010]")
        ax.set_title(name)
    figure.legend(*np.atleast_1d(row)[0].get_legend_handles_labels(), loc="outside lower center", ncol=len(MODES),
                  markerscale=8, frameon=False, fontsize=9)
    return figure
