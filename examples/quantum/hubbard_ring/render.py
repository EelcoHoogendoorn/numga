"""The low levels of the Hubbard ring against the Heisenberg ring, and a flipped spin walking the ring."""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from examples.quantum.hubbard_ring import core
from examples.util import charts
from examples.util.wgsl import planar, simplices
from examples.util.wgsl.camera import flat_panel
from numga import stack

UP, DOWN, LEVELS = "#c0392b", "#2e86c1", "#7d3c98"
# The rim's ink and the sites'.
RIM, SITE = np.full(3, 0.85), np.full(3, 0.3)
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
    # The counts are numbers of the chart, read out of the ring's algebra to set the bars' heights.
    counts = occupation.to_array()                                             # [cases, times, sites, spins]
    tips = places[:, None] + mv.y.dual() * counts * (spin_height * np.array([1, -1]))   # [cases, times, sites, spins] Point-
    # Each bar from its site to its tip, up for spin up and down for spin down.
    bars = stack((places[:, None].broadcast_to(tips.shape), tips), axis=-1)       # [cases, times, sites, spins, 2] Point-
    n_bars = np.prod(tips.shape[2:])
    bar_colours = np.tile(np.stack((to_rgb(UP), to_rgb(DOWN))), (len(places), 1))
    labels = [f"$U/t = {ratio:g}$" for ratio in ratios]
    with charts.style():
        # A ring of four sites in each panel, about 200 pixels across, over its label.
        figure = plt.figure(figsize=(ring_pixels * len(labels) / charts.DPI, (ring_pixels + label_pixels) / charts.DPI))
        turns = (mv.xy * (-np.linspace(0, 2 * np.pi, n_rim_samples) / 2)).exp()
        rim = planar.origin + (turns >> (mv.x.dual() * rim_radius))           # [rim samples] Point-
        rings = []
        for case, label in enumerate(labels):
            ax = figure.add_axes((case / len(labels), label_pixels / (ring_pixels + label_pixels), 1 / len(labels),
                                  ring_pixels / (ring_pixels + label_pixels)))
            ax.set(xlim=(-1.3, 1.3), ylim=(-1.3, 1.3), aspect="equal")
            camera = flat_panel(ax, planar.ga)
            outline = simplices.strokes(rim, simplices.Mesh.chains([n_rim_samples]), RIM, charts.FRAME_PX, camera)
            sites = simplices.markers(places, SITE, charts.MARKER_PX / 2, camera)
            rings.append((camera, outline, sites, simplices.overlay(ax, [outline, sites])))
            # The ring's label stays, its ticks and frame gone.
            ax.set_axis_on()
            ax.set(xticks=[], yticks=[])
            ax.set_xlabel(label)
            for side in ax.spines.values():
                side.set_visible(False)
        for index in range(tips.shape[1]):
            for (camera, outline, sites, shown), case_bars in zip(rings, bars[:, index]):
                # The spin counts as bars about as wide as the sites' dots, under the sites.
                spins = simplices.strokes(case_bars.reshape(-1), simplices.Mesh.chains([2] * n_bars), bar_colours,
                                          spin_pixels, camera, cap=simplices.Cap.BUTT)
                shown.set_data(simplices.composited([outline, spins, sites]))
            figure.canvas.draw()
            yield np.asarray(figure.canvas.buffer_rgba())[..., :3].copy()
        plt.close(figure)
