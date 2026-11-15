"""Phonon-focusing images and wave-surface sections read out for matplotlib."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D

from numga import stack
from examples.util import charts
from examples.mechanics.crystal_waves import core
from examples.mechanics.crystal_waves.scenarios import modes
from numga.topology import Topology
from numga.render import Camera, simplices
from examples.util.panels import offset_panel, overlay, pinned, room

mode_colours = ("#c0392b", "#2e86c1", "#7d3c98")
cross_colour, label_colour = "0.85", "0.4"
# Margin around the wave fronts as a fraction of their maximum reach, leaving room for axis labels.
margin_share = 0.45
# Span of the cube face shown in the focusing panels.
face_span = (-1.5, 1.5)
# Diameter of the wave-front markers in pixels.
dot_px = 2


def maximum_reach(directions: core.Direction) -> float:
    """Largest absolute x or y coordinate across `directions`, read by pairing with the first two cube planes."""
    return float((core.axes[:2, None] & directions.reshape(-1)).abs().max().to_array())


def cross_labels(ax: plt.Axes, limit: float, x_label: str, y_label: str) -> None:
    """Configures a square axisless panel of half-width `limit` with crystal-direction labels at the cross ends."""
    ax.text(limit, 0, x_label, ha="right", va="top", color=label_colour)
    ax.text(0, limit, f" {y_label}", ha="left", va="top", color=label_colour)
    ax.set(xlim=(-limit, limit), ylim=(-limit, limit), aspect="equal")
    ax.set_axis_off()


def cross_image(limit: float, camera: Camera) -> np.ndarray:
    """Renders a faint reference cross of half-width `limit` through the origin."""
    x_plane, y_plane, z_plane = core.axes                           # Plane+, Plane+, Plane+
    ends = limit * stack((-x_plane.dual(), x_plane.dual(), -y_plane.dual(), y_plane.dual()))   # [4] Direction-
    return simplices.strokes(
        ends, Topology.chains([2, 2]), np.array(to_rgb(cross_colour)), charts.FRAME_PX, camera, cap=simplices.Cap.BUTT
    )


@charts.style()
def focusing_figure(velocities: core.Direction, names: list[str]) -> plt.Figure:
    """Phonon-focusing intensity panels across materials (rows) and wave modes (columns) on the cube face z = 1."""
    n_materials, n_modes = len(names), len(modes)
    figure, grid = charts.subplots(200 * n_modes, 220 * n_materials, nrows=n_materials, ncols=n_modes)
    for row, name in zip(np.atleast_2d(grid), names):
        for ax, mode in zip(row, modes):
            ax.set_title(f"{name}: {mode}")
            ax.set_axis_off()
    figure.tight_layout()
    for row, velocity in zip(np.atleast_2d(grid), velocities):
        for ax, wave in zip(row, (velocity[:, mode] for mode in range(n_modes))):
            # Select rays heading toward the z = 1 cube face and normalize each ray to unit z reading.
            forward_rays = wave[(core.axes[2] & wave) > 0]          # [...] Direction-
            landings = forward_rays / (core.axes[2] & forward_rays)   # [...] Direction-
            face_coordinates = (core.axes[:2] & landings[:, None]).to_array()   # [..., 2]
            n_bins = min(room(ax))
            density, _, _ = np.histogram2d(
                face_coordinates[:, 0], face_coordinates[:, 1], bins=n_bins, range=[face_span, face_span]
            )
            pinned(ax, np.log1p(density.T[::-1]), (*face_span, *face_span)).set_cmap("inferno")
    return figure


@charts.style()
def wave_fronts_figure(velocities: core.Direction, names: list[str]) -> plt.Figure:
    """Wave-surface cross-sections in the z = 0 cube face, one panel per material."""
    n_materials, n_modes = len(names), len(modes)
    figure, row = charts.subplots(250 * n_materials, 300, ncols=n_materials, layout="constrained")
    limits = [(1 + margin_share) * maximum_reach(velocity) for velocity in velocities]
    for ax, limit, name in zip(np.atleast_1d(row), limits, names):
        cross_labels(ax, limit, "[100]", "[010]")
        ax.set_title(name)
    handles = [
        Line2D([], [], linestyle="none", marker="o", markersize=charts.MARKER_SIZE, color=colour, label=mode)
        for mode, colour in zip(modes, mode_colours)
    ]
    figure.legend(handles=handles, loc="outside lower center", ncol=n_modes, frameon=False)
    for ax, velocity, limit in zip(np.atleast_1d(row), velocities, limits):
        camera = offset_panel(ax, core.mv.x, core.mv.y)
        overlay(
            ax,
            [cross_image(limit, camera)]
            + [
                simplices.markers(velocity[:, mode], np.array(to_rgb(colour)), dot_px / 2, camera)
                for mode, colour in zip(range(n_modes), mode_colours)
            ],
        )
    return figure
