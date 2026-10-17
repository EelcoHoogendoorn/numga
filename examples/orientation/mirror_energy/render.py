"""A mass distribution and its mirror, with tangent velocities on the unit sphere."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.artist import Artist

from examples.animation import capture
from examples.orientation.mirror_energy.core import Point, ga

COLOURS = ("#df7246", "#238c9b", "#7662a1")


# --- plumbing -------------------------------------------------------------------------
def _scene(paths: np.ndarray) -> tuple[plt.Figure, list[plt.Axes]]:
    figure = plt.figure(figsize=(9, 4), dpi=80)
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0)
    axes = []
    angle = np.linspace(0, 2 * np.pi, 80)
    for case in range(paths.shape[1]):
        ax = figure.add_subplot(1, 2, case + 1, projection="3d", computed_zorder=False)
        ax.set(xlim=(-1.2, 1.2), ylim=(-1.2, 1.2), zlim=(-1.2, 1.2))
        ax.set_box_aspect((1, 1, 1), zoom=1.1)
        ax.view_init(elev=23, azim=-63)
        ax.set_axis_off()
        # Great circles provide depth without hiding the motion behind a solid shell.
        zeros = np.zeros_like(angle)
        for ring in ((np.cos(angle), np.sin(angle), zeros),
                     (np.cos(angle), zeros, np.sin(angle)),
                     (zeros, np.cos(angle), np.sin(angle))):
            ax.plot(*ring, color="#aab4bf", alpha=0.55, linewidth=0.8, zorder=0)
        closed = np.concatenate([paths[:, case], paths[:1, case]], axis=0)
        for mass, colour in enumerate(COLOURS):
            ax.plot(*closed[:, mass].T, color=colour, alpha=0.5, linewidth=1.4, zorder=1)
        axes.append(ax)
    return figure, axes


def _markers(axes: list[plt.Axes], positions: np.ndarray, velocities: np.ndarray,
             masses: np.ndarray) -> list[Artist]:
    artists = []
    for case, ax in enumerate(axes):
        # Joining the masses makes the reflected shape visible throughout the turn.
        closed = np.concatenate([positions[case], positions[case, :1]], axis=0)
        artists.extend(ax.plot(*closed.T, color="#75808a", linewidth=1.2, alpha=0.8, zorder=2))
        artists.append(ax.scatter(*positions[case].T, s=70 * masses, c=COLOURS,
                                  edgecolors="white", linewidths=0.8, depthshade=False, zorder=4))
        artists.append(ax.quiver(*positions[case].T, *velocities[case].T, length=0.35,
                                 color=COLOURS, linewidth=2, arrow_length_ratio=0.3, zorder=3))
    return artists


def draw_motion(positions: Point, velocities: Point, masses: np.ndarray) -> plt.Figure:
    paths = positions.cast(ga.subspace("yz zx xy")).kernel
    tangents = velocities.cast(ga.subspace("yz zx xy")).kernel
    figure, axes = _scene(paths)
    _markers(axes, paths[0], tangents[0], masses)
    return figure


def animate_motion(positions: Point, velocities: Point, masses: np.ndarray) -> list[np.ndarray]:
    paths = positions.cast(ga.subspace("yz zx xy")).kernel
    tangents = velocities.cast(ga.subspace("yz zx xy")).kernel
    figure, axes = _scene(paths)
    frames = []
    for position, velocity in zip(paths, tangents):
        artists = _markers(axes, position, velocity, masses)
        frames.append(capture(figure))
        for artist in artists:
            artist.remove()
    plt.close(figure)
    return frames
