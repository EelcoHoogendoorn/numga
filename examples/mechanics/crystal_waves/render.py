"""Phonon-focusing images and wave-surface sections: vectors read out for matplotlib."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.util import charts
from examples.mechanics.crystal_waves import core
from examples.mechanics.crystal_waves.scenarios import MODES
from examples.util.wgsl import pinned, room

MODE_COLOURS = ("#c0392b", "#2e86c1", "#7d3c98")
CROSS_COLOUR, LABEL_COLOUR = "0.85", "0.4"
# The margin around the wave fronts, as a share of their reach: room for the names of the cross at its ends.
MARGIN = 0.45
# The span of the cube face the focusing panels show, across and up.
FACE = (-1.5, 1.5)
# The size of the wave fronts' point markers in pixels, each dot drawn half as wide.
DOT_PX = 3


def components(directions: core.Antivector) -> np.ndarray:
    """Euclidean components of directions, read by the planes across the axes, on the last axis."""
    return (core.axes & directions[..., None]).to_array()


def cross(ax: plt.Axes, limit: float, across: str, up: str) -> None:
    """A square plane of half width `limit` without scales: a faint cross through the origin, its lines named at
    their ends."""
    # The cross as thin as a chart's frame, which it stands in for.
    ax.axhline(0, color=CROSS_COLOUR, linewidth=charts.FRAME_WIDTH, zorder=0)
    ax.axvline(0, color=CROSS_COLOUR, linewidth=charts.FRAME_WIDTH, zorder=0)
    ax.text(limit, 0, across, ha="right", va="top", color=LABEL_COLOUR)
    ax.text(0, limit, f" {up}", ha="left", va="top", color=LABEL_COLOUR)
    ax.set(xlim=(-limit, limit), ylim=(-limit, limit), aspect="equal")
    ax.set_axis_off()


@charts.style()
def draw_focusing(velocities: core.Vector, names: list[str]) -> plt.Figure:
    """One row per crystal, one column per wave: how densely the energy of evenly spread headings
    lands on a cube face, seen from a source at the centre of the opposite face, binned on the pixels of each
    panel."""
    # Three quarters of the column: a square image a crystal and a wave, each under its title.
    figure, grid = charts.subplots(200 * len(MODES), 220 * len(names), nrows=len(names), ncols=len(MODES))
    for row, name in zip(np.atleast_2d(grid), names):
        for ax, mode in zip(row, MODES):
            ax.set_title(f"{name}: {mode}")
            ax.set_axis_off()
    figure.tight_layout()
    for row, velocity in zip(np.atleast_2d(grid), velocities):
        for ax, wave in zip(row, (velocity[:, mode] for mode in range(len(MODES)))):
            # The rays that head for the face one unit along z, and where each meets it: the ray's direction scaled
            # to unit reading along z, read across the face.
            ahead = wave[(core.axes[2] & wave) > 0]                    # [rays] Antivector-
            landing = ahead / (core.axes[2] & ahead)                   # [rays] Antivector-
            face = (core.axes[:2] & landing[:, None]).to_array()       # [rays, 2]
            bins = min(room(ax))
            density, _, _ = np.histogram2d(face[:, 0], face[:, 1], bins=bins, range=[FACE, FACE])
            pinned(ax, np.log1p(density.T[::-1]), (*FACE, *FACE)).set_cmap("inferno")
    return figure


@charts.style()
def draw_wave_fronts(velocities: core.Vector, names: list[str]) -> plt.Figure:
    """One panel per crystal: where each wave's energy is after unit time, in the cube face, in the directions of
    the crystal."""
    # Nearly the whole column: a panel a crystal, each a dense section of three wave surfaces, over one legend.
    figure, row = charts.subplots(250 * len(names), 300, ncols=len(names), layout="constrained")
    for ax, velocity, name in zip(np.atleast_1d(row), velocities, names):
        flow = components(velocity)                                    # [count, waves, 3]
        for wave, mode, colour in zip(np.moveaxis(flow, 1, 0), MODES, MODE_COLOURS):
            # Small dots, so that the sampled headings read as curves.
            ax.plot(wave[:, 0], wave[:, 1], ".", markersize=charts.points(DOT_PX), color=colour, label=mode)
        cross(ax, (1 + MARGIN) * np.abs(flow[..., :2]).max(), "[100]", "[010]")
        ax.set_title(name)
    # The legend's dots as wide as a chart's markers.
    figure.legend(*np.atleast_1d(row)[0].get_legend_handles_labels(), loc="outside lower center", ncol=len(MODES),
                  markerscale=2 * charts.MARKER_PX / DOT_PX, frameon=False)
    return figure
