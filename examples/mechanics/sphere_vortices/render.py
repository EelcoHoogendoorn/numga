"""Drawing for vortices on the sphere, in the projective picture: every vortex at both of its antipodal
points, the pair its bivector stands for. Each sphere is seen orthographically,
so that its outline is a circle; a shaded disc for the sphere, the points as dots with fading trails
behind them. What lies on the far side is left out: its antipode on the near side shows it.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba_array

from examples.animation import capture
from examples.mechanics.sphere_vortices import core

# One colour per vortex, repeating; the sphere's colour, its light's direction on the screen and the share
# of light that falls everywhere; the trail's length in
# steps, the line width and dot size; the shading's
# resolution in pixels, and each sphere's size in inches.
STRANDS = np.array(["#c84b31", "#2d6a9f", "#4c956c", "#e0a458", "#8e7dbe", "#d17a9b", "#5fa8d3", "#7f6a4f"])
SPHERE_COLOUR = np.array([0.90, 0.91, 0.93])
LIGHT, AMBIENT = np.array([-0.4, 0.5, 0.77]), 0.55
TRAIL = 240
WIDTH, DOT = 1.6, 28
RESOLUTION = 300
SIZE = 3.0


def coordinates(pivots: core.Pivot) -> np.ndarray:
    """The unit directions `[..., 3]` of points of the sphere: where they lie on it."""
    parts = pivots.cast(core.ga.subspace("yz zx xy")).kernel
    return parts / np.linalg.norm(parts, axis=-1, keepdims=True)


def screen(view: tuple[float, float]) -> np.ndarray:
    """The right, up and toward-the-viewer directions `[3, 3]` of a view from (elevation, azimuth) in
    degrees."""
    elevation, azimuth = np.radians(view[0]), np.radians(view[1])
    toward = np.array([np.cos(elevation) * np.cos(azimuth), np.cos(elevation) * np.sin(azimuth), np.sin(elevation)])
    right = np.cross([0.0, 0.0, 1.0], toward)
    right /= np.linalg.norm(right)
    return np.stack([right, np.cross(toward, right), toward])


def shading() -> np.ndarray:
    """The sphere as an RGBA image, lit from `LIGHT`, transparent outside its outline."""
    across = np.linspace(-1, 1, RESOLUTION)
    right, up = np.meshgrid(across, across[::-1])
    inside = right**2 + up**2 < 1
    normals = np.stack([right, up, np.sqrt(np.clip(1 - right**2 - up**2, 0, 1))], axis=-1)
    light = AMBIENT + (1 - AMBIENT) * np.clip(normals @ (LIGHT / np.linalg.norm(LIGHT)), 0, 1)
    return np.concatenate([SPHERE_COLOUR * light[..., None], inside[..., None].astype(float)], axis=-1)


def animate(histories: list[core.Pivot], view: tuple[float, float], every: int) -> list[np.ndarray]:
    """Frames of vortices on spheres side by side, one sphere per history `[steps, vortices]`, all of one
    length, seen from (elevation, azimuth) in degrees, every vortex at both its antipodal points: a frame
    every `every` steps, its trails through every step."""
    frame = screen(view)
    seen = [coordinates(pivots) @ frame.T for pivots in histories]              # [times, vortices, 3] each
    # Each vortex and its antipode, in one colour.
    seen = [np.concatenate([places, -places], axis=1) for places in seen]       # [times, 2 vortices, 3] each
    disc = shading()
    images = []
    for time in range(0, len(seen[0]), every):
        figure = plt.figure(figsize=(SIZE * len(seen), SIZE), dpi=90)
        for index, places in enumerate(seen):
            ax = figure.add_axes((index / len(seen), 0, 1 / len(seen), 1))
            ax.imshow(disc, extent=(-1, 1, -1, 1), zorder=0)
            colours = np.tile(np.resize(STRANDS, places.shape[1] // 2), 2)       # [2 vortices]
            # The trail of the last steps on the near side, older parts fainter.
            first = max(time - TRAIL, 0)
            trail = places[first:time + 1]                                      # [trail, 2 vortices, 3]
            segments = np.stack([trail[:-1, :, :2], trail[1:, :, :2]], axis=-2).reshape(-1, 2, 2)
            age = np.broadcast_to(np.linspace(0, 1, len(trail))[1:, None], (len(trail) - 1, places.shape[1])).reshape(-1)
            near = (trail[1:, :, 2] > 0).reshape(-1)
            tints = to_rgba_array(np.broadcast_to(colours, (len(trail) - 1, places.shape[1])).reshape(-1))
            tints[:, 3] = age
            ax.add_collection(LineCollection(segments[near], colors=tints[near], linewidths=WIDTH, capstyle="round", zorder=1))
            front = places[time, :, 2] > 0                                      # [2 vortices]
            ax.scatter(places[time, front, 0], places[time, front, 1], c=colours[front], s=DOT, linewidths=0, zorder=2)
            ax.set(xlim=(-1.08, 1.08), ylim=(-1.08, 1.08))
            ax.set_aspect("equal")
            ax.set_axis_off()
        images.append(capture(figure))
        plt.close(figure)
    return images
