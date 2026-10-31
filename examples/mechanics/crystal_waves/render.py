"""Phonon-focusing images and wave-surface sections: vectors read out for matplotlib."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D

from numga import stack
from examples.util import charts
from examples.mechanics.crystal_waves import core
from examples.mechanics.crystal_waves.scenarios import MODES
from examples.util.wgsl import pinned, room, simplices
from examples.util.wgsl.camera import Camera, flat_panel

MODE_COLOURS = ("#c0392b", "#2e86c1", "#7d3c98")
CROSS_COLOUR, LABEL_COLOUR = "0.85", "0.4"
# The margin around the wave fronts, as a share of their reach: room for the names of the cross at its ends.
MARGIN = 0.45
# The span of the cube face the focusing panels show, across and up.
FACE = (-1.5, 1.5)
# The wave fronts' dots, in pixels across.
DOT_PX = 2


def farthest(directions: core.Antivector) -> float:
    """The farthest any direction reaches across x or y, read by the planes across those axes."""
    return float((core.axes[:2, None] & directions.reshape(-1)).abs().max().to_array())


def cross(ax: plt.Axes, limit: float, across: str, up: str) -> None:
    """A square plane of half width `limit` without scales, the names of the lines of its cross through the origin at
    their ends."""
    ax.text(limit, 0, across, ha="right", va="top", color=LABEL_COLOUR)
    ax.text(0, limit, f" {up}", ha="left", va="top", color=LABEL_COLOUR)
    ax.set(xlim=(-limit, limit), ylim=(-limit, limit), aspect="equal")
    ax.set_axis_off()


def cross_drawn(limit: float, camera: Camera) -> np.ndarray:
    """The faint cross through the origin, out to `limit` along x and y either way, as thin as a chart's frame, which
    it stands in for: its picture."""
    x, y, _ = core.axes
    ends = limit * stack((-x.dual(), x.dual(), -y.dual(), y.dual()))         # [4] Antivector-
    return simplices.strokes(ends, simplices.Mesh.chains([2, 2]), np.array(to_rgb(CROSS_COLOUR)), charts.FRAME_PX,
                             camera, cap=simplices.Cap.BUTT)


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
    limits = [(1 + MARGIN) * farthest(velocity) for velocity in velocities]
    for ax, limit, name in zip(np.atleast_1d(row), limits, names):
        cross(ax, limit, "[100]", "[010]")
        ax.set_title(name)
    # The legend's dots as wide as a chart's markers.
    handles = [Line2D([], [], linestyle="none", marker="o", markersize=charts.MARKER_SIZE, color=colour, label=mode)
               for mode, colour in zip(MODES, MODE_COLOURS)]
    figure.legend(handles=handles, loc="outside lower center", ncol=len(MODES), frameon=False)
    for ax, velocity, limit in zip(np.atleast_1d(row), velocities, limits):
        camera = flat_panel(ax, core.ga)
        # Small dots, so that the sampled headings read as curves, each wave over the slower ones.
        simplices.overlay(ax, [cross_drawn(limit, camera)] + [
            simplices.markers(velocity[:, mode], np.array(to_rgb(colour)), DOT_PX / 2, camera)
            for mode, colour in zip(range(len(MODES)), MODE_COLOURS)])
    return figure
