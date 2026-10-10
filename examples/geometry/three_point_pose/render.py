"""Marker triangles before, during and after geometric pose reconstruction."""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from examples.animation import capture
from examples.geometry.three_point_pose import core

COLOUR = "#218c9d"
TARGET = "#ce713f"
ORDER = [0, 1, 2, 0]


# --- plumbing -------------------------------------------------------------------------
def coordinates(points: core.Point) -> np.ndarray:
    values = points.cast(core.ga.subspace("yzw zxw xyw zyx")).kernel
    return values[..., :3] / values[..., 3:]


def frame(ax: plt.Axes, source: np.ndarray, target: np.ndarray) -> None:
    bounds = np.concatenate([source, target])
    centre = (bounds.min(axis=0) + bounds.max(axis=0)) / 2
    radius = np.max(bounds.max(axis=0) - bounds.min(axis=0)) * 0.6
    ax.set(xlim=(centre[0] - radius, centre[0] + radius),
           ylim=(centre[1] - radius, centre[1] + radius),
           zlim=(centre[2] - radius, centre[2] + radius))
    ax.set_box_aspect((1, 1, 1), zoom=1.5)
    ax.view_init(elev=55, azim=-110)
    ax.set_axis_off()
    ax.plot(*target[ORDER].T, color=TARGET, linewidth=2, linestyle="--")
    ax.scatter(*target.T, color=TARGET, s=65, depthshade=False)


def triangle(ax: plt.Axes, points: np.ndarray) -> Poly3DCollection:
    artist = Poly3DCollection([points], facecolors=COLOUR, edgecolors=COLOUR,
                              alpha=0.45, linewidths=2)
    ax.add_collection3d(artist)
    return artist


def posed(source: core.Point, target: core.Point, placed: core.Point) -> tuple[plt.Figure, Poly3DCollection]:
    figure = plt.figure(figsize=(6, 4.5), layout="constrained")
    ax = figure.add_subplot(projection="3d")
    frame(ax, coordinates(source), coordinates(target))
    return figure, triangle(ax, coordinates(placed))


def draw_pose(source: core.Point, target: core.Point, placed: core.Point) -> plt.Figure:
    figure, _ = posed(source, target, placed)
    return figure


def draw_stages(source: core.Point, target: core.Point, stages: Iterator[core.Point]) -> plt.Figure:
    stages = tuple(stages)
    figure = plt.figure(figsize=(11, 3.4), layout="constrained")
    for index, placed in enumerate(stages):
        ax = figure.add_subplot(1, len(stages), index + 1, projection="3d")
        frame(ax, coordinates(source), coordinates(target))
        triangle(ax, coordinates(placed))
    return figure


def animate(states: Iterator[core.Point], source: core.Point, target: core.Point) -> list[np.ndarray]:
    figure, artist = posed(source, target, source)
    frames = []
    for state in states:
        artist.set_verts([coordinates(state)])
        frames.append(capture(figure))
    plt.close(figure)
    return frames
