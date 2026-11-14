"""Line plots and ring panels display the Hubbard ring's low-energy spin spectrum and site spin occupations."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from examples.quantum.hubbard_ring import core
from examples.util import charts
from examples.util.panels import overlay, panel
from numga import stack
from numga.render import drawing, planar, simplices
from numga.topology import Topology

up_colour, down_colour, level_colour = "#c0392b", "#2e86c1", "#7d3c98"
rim_colour, site_colour = np.full(3, 0.85), np.full(3, 0.3)
# Four ring sites ordered counterclockwise starting at the bottom left.
mv = planar.mv
axes = stack((mv.x, mv.y))                                                   # [2] Plane+
site_points = planar.origin + (axes.dual() * (np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * 0.7)).sum(axis=-1)   # [4] Point-
n_rim_samples = 119
rim_radius = 0.99
spin_height = 0.5
ring_pixels, label_pixels, spin_pixels = 200, 24, 4


@charts.style()
def levels_figure(ratios: np.ndarray, excitations: core.Scalar, heisenberg_levels: np.ndarray) -> plt.Figure:
    """Level curves of `excitations` (`[n_samples, n_spin_levels] Scalar+`) against repulsion-to-hopping
    `ratios` (`[n_samples]`) approach the dashed `heisenberg_levels`."""
    figure, ax = charts.subplots(400, 260, layout="constrained")
    ax.plot(ratios, excitations.to_array(), color=level_colour)
    for level in heisenberg_levels:
        ax.axhline(level, color="0.5", linestyle="--")
    ax.set(xlim=(ratios[0], ratios[-1]), ylim=(0, heisenberg_levels[-1] + 0.6), xticks=[],
           yticks=np.arange(heisenberg_levels[-1] + 1),
           xlabel="repulsion / hopping", ylabel="excitation / exchange")
    ax.spines[["top", "right"]].set_visible(False)
    return figure


def ring_frames(ratios: np.ndarray, occupations: Iterable[core.Scalar]) -> Iterator[np.ndarray]:
    """Yields RGB frames showing each site's spin-up occupation as a red upward bar and its spin-down
    occupation as a blue downward bar across `ratios` (`[n_cases]`)."""
    n_cases = len(ratios)
    n_bars = 8
    bar_colours = np.tile(np.stack((to_rgb(up_colour), to_rgb(down_colour))), (4, 1))
    labels = [f"repulsion / hopping = {ratio:g}" for ratio in ratios]
    with charts.style():
        figure = plt.figure(figsize=(ring_pixels * n_cases / charts.DPI, (ring_pixels + label_pixels) / charts.DPI))
        rim_turns = (mv.xy * (-np.linspace(0, 2 * np.pi, n_rim_samples, endpoint=False) / 2)).exp()   # [n_rim_samples] Rotor+
        rim_offsets = rim_turns >> (mv.x.dual() * rim_radius)                                         # [n_rim_samples] Point-
        rim = planar.origin + rim_offsets                                                             # [n_rim_samples] Point-
        rings = []
        for case_index, label in enumerate(labels):
            ax = figure.add_axes((case_index / n_cases, label_pixels / (ring_pixels + label_pixels), 1 / n_cases,
                                  ring_pixels / (ring_pixels + label_pixels)))
            ax.set(xlim=(-1.3, 1.3), ylim=(-1.3, 1.3), aspect="equal")
            camera = panel(ax, planar.mv.x, planar.mv.y, planar.mv.w)
            outline = simplices.strokes(rim, Topology.loops([n_rim_samples]), rim_colour, charts.FRAME_PX, camera)
            sites = simplices.markers(site_points, site_colour, charts.MARKER_PX / 2, camera)
            rings.append((camera, outline, sites, overlay(ax, [outline, sites])))
            ax.set_axis_on()
            ax.set(xticks=[], yticks=[])
            ax.set_xlabel(label)
            for spine in ax.spines.values():
                spine.set_visible(False)
        try:
            for occupation in occupations:
                counts = occupation.to_array()                                                        # [n_cases, 4, 2]
                tips = site_points[:, None] + mv.y.dual() * counts * (spin_height * np.array([1, -1]))   # [n_cases, 4, 2] Point-
                bars = stack((site_points[:, None].broadcast_to(tips.shape), tips), axis=-1)          # [n_cases, 4, 2, 2] Point-
                for (camera, outline, sites, shown), case_bars in zip(rings, bars):
                    spins = simplices.strokes(case_bars.reshape(-1), Topology.chains([2] * n_bars), bar_colours,
                                              spin_pixels, camera, cap=simplices.Cap.BUTT)
                    shown.set_data(drawing.composited([outline, spins, sites]))
                figure.canvas.draw()
                yield np.asarray(figure.canvas.buffer_rgba())[..., :3].copy()
        finally:
            plt.close(figure)
