"""The low levels of the Hubbard ring against the Heisenberg ring, and a flipped spin walking the ring."""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.pyplot as plt
import numpy as np

from examples.quantum.hubbard_ring import core
from examples.util import charts

UP, DOWN, LEVELS = "#c0392b", "#2e86c1", "#7d3c98"
# The sites on a circle, the first at the bottom left, in the order the hops go round.
PLACES = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]) * 0.7
# The side of each ring's panel and the room for its label below it, and the width of the spins' bars, in pixels.
RING_PX, LABEL_PX, SPIN_PX = 200, 24, 4


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
    counts = occupation.to_array()                                            # [cases, times, sites, spins]
    labels = [f"$U/t = {ratio:g}$" for ratio in ratios]
    with charts.style():
        # A ring of four sites in each panel, about 200 pixels across, over its label.
        figure = plt.figure(figsize=(RING_PX * len(labels) / charts.DPI, (RING_PX + LABEL_PX) / charts.DPI))
        turn = np.linspace(0, 2 * np.pi, 120)
        arrows = []
        for case, label in enumerate(labels):
            ax = figure.add_axes((case / len(labels), LABEL_PX / (RING_PX + LABEL_PX), 1 / len(labels),
                                  RING_PX / (RING_PX + LABEL_PX)))
            ax.plot(0.99 * np.cos(turn), 0.99 * np.sin(turn), color="0.85", linewidth=charts.FRAME_WIDTH)
            ax.scatter(*PLACES.T, color="0.3", zorder=3)
            # The spin counts as bars about as wide as the sites' dots.
            ups = [ax.plot([], [], color=UP, linewidth=charts.points(SPIN_PX), solid_capstyle="butt")[0] for _ in PLACES]
            downs = [ax.plot([], [], color=DOWN, linewidth=charts.points(SPIN_PX), solid_capstyle="butt")[0]
                     for _ in PLACES]
            arrows.append((ups, downs))
            ax.set(xlim=(-1.3, 1.3), ylim=(-1.3, 1.3), xticks=[], yticks=[])
            ax.set_xlabel(label)
            ax.set_aspect("equal")
            for side in ax.spines.values():
                side.set_visible(False)
        for index in range(counts.shape[1]):
            for (ups, downs), case in zip(arrows, counts[:, index]):
                for up, down, (x, y), (up_count, down_count) in zip(ups, downs, PLACES, case):
                    up.set_data([x, x], [y, y + 0.5 * up_count])
                    down.set_data([x, x], [y, y - 0.5 * down_count])
            figure.canvas.draw()
            yield np.asarray(figure.canvas.buffer_rgba())[..., :3].copy()
        plt.close(figure)
