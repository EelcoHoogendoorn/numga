"""The low levels of the Hubbard ring against the Heisenberg ring, and a flipped spin walking the ring."""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.pyplot as plt
import numpy as np

from examples.quantum.hubbard_ring import core
from examples.util import charts
from examples.util.wgsl import planar
from numga import stack

UP, DOWN, LEVELS = "#c0392b", "#2e86c1", "#7d3c98"
# The sites on a circle, the first at the bottom left, in the order the hops go round.
mv = planar.mv
axes = stack((mv.x, mv.y))                                                   # [2] Plane+
places = planar.origin + (axes.dual() * (np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * 0.7)).sum(axis=-1)
n_rim_samples = 120
rim_radius = 0.99
spin_height = 0.5
# The side of each ring's panel and the room for its label below it, and the width of the spins' bars, in pixels.
ring_pixels, label_pixels, spin_pixels = 200, 24, 4


# --- plumbing -------------------------------------------------------------------------
@charts.style()
def draw_levels(ratios: np.ndarray, excitations: core.Scalar, heisenberg_levels: np.ndarray) -> plt.Figure:
    """The lowest levels above the ground state, in units of the exchange, against the repulsion; the
    Heisenberg ring's levels dashed."""
    # Half the column: a band of level curves against a few dashed levels, and the height the vertical label needs.
    figure, ax = charts.subplots(400, 260, layout="constrained")
    ax.plot(ratios, excitations.to_array(), color=LEVELS)
    for level in heisenberg_levels:
        ax.axhline(level, color="0.5", linestyle="--")
    # The levels read off where the Heisenberg ring puts them, in units of the exchange.
    ax.set(xlim=(ratios[0], ratios[-1]), ylim=(0, heisenberg_levels[-1] + 0.6), xticks=[], yticks=np.arange(heisenberg_levels[-1] + 1),
           xlabel=r"repulsion / hopping, $U/t$", ylabel=r"excitation, $\Delta E\,U/4t^2$")
    ax.spines[["top", "right"]].set_visible(False)
    return figure


def animate_ring(ratios: np.ndarray, occupation: core.Scalar) -> Iterator[np.ndarray]:
    """Each site's spin-up count as an arrow up and its spin-down count as an arrow down, one ring per
    ratio of repulsion to hopping, frame by frame."""
    counts = planar.context.include_from_algebra(occupation)                                  # [cases, times, sites, spins] Scalar+
    tips = places[:, None] + mv.y.dual() * counts * (spin_height * np.array([1, -1]))
    locations = (axes & places[..., None]).to_array()
    endpoints = (axes & tips[..., None]).to_array()                           # [cases, times, sites, spins, 2]
    labels = [f"$U/t = {ratio:g}$" for ratio in ratios]
    with charts.style():
        # A ring of four sites in each panel, about 200 pixels across, over its label.
        figure = plt.figure(figsize=(ring_pixels * len(labels) / charts.DPI, (ring_pixels + label_pixels) / charts.DPI))
        turns = (mv.xy * (-np.linspace(0, 2 * np.pi, n_rim_samples) / 2)).exp()
        rim = planar.origin + (turns >> (mv.x.dual() * rim_radius))
        outline = (axes & rim[..., None]).to_array()
        arrows = []
        for case, label in enumerate(labels):
            ax = figure.add_axes((case / len(labels), label_pixels / (ring_pixels + label_pixels), 1 / len(labels),
                                  ring_pixels / (ring_pixels + label_pixels)))
            ax.plot(*outline.T, color="0.85", linewidth=charts.FRAME_WIDTH)
            ax.scatter(*locations.T, color="0.3", zorder=3)
            # The spin counts as bars about as wide as the sites' dots.
            ups = [ax.plot([], [], color=UP, linewidth=charts.points(spin_pixels), solid_capstyle="butt")[0] for _ in places]
            downs = [ax.plot([], [], color=DOWN, linewidth=charts.points(spin_pixels), solid_capstyle="butt")[0]
                     for _ in places]
            arrows.append((ups, downs))
            ax.set(xlim=(-1.3, 1.3), ylim=(-1.3, 1.3), xticks=[], yticks=[])
            ax.set_xlabel(label)
            ax.set_aspect("equal")
            for side in ax.spines.values():
                side.set_visible(False)
        for index in range(tips.shape[1]):
            for (ups, downs), case in zip(arrows, endpoints[:, index]):
                for up, down, start, (up_tip, down_tip) in zip(ups, downs, locations, case):
                    up.set_data(*np.stack((start, up_tip)).T)
                    down.set_data(*np.stack((start, down_tip)).T)
            figure.canvas.draw()
            yield np.asarray(figure.canvas.buffer_rgba())[..., :3].copy()
        plt.close(figure)
