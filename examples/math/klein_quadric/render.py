"""Drawing for the four-line problem: every line as its chord of a ball about the origin; the lines
on the hyperboloid faint, the three lines that fix it dark, the fourth orange, and the transversals
red where they are real."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from examples.animation import capture
from examples.math.klein_quadric import core

mv = core.mv
# Lines are drawn inside this ball.
RADIUS = 3.0
AXES = mv("x y z", np.eye(3))                                                   # [3] Plane


def real(kernel: np.ndarray) -> np.ndarray:
    """Which extensors, by their coefficients on the last axis, are real up to round-off."""
    return np.abs(kernel.imag).max(axis=-1) <= 1e-6 * np.abs(kernel).max(axis=-1)


def coordinates(points: core.Point) -> np.ndarray:
    """Euclidean coordinates of finite points: their pairings with the coordinate planes, at unit
    weight."""
    return ((AXES & points[..., None]) / (mv.w & points[..., None])).to_array().real


def chords(lines: core.Bivector) -> np.ndarray:
    """The two ends of each line's chord of the ball, NaN for a line that misses it."""
    unit = lines.normalized()
    nearest = coordinates((core.ORIGIN | unit) ^ unit)                        # [..., 3]
    heading = (AXES & (unit ^ mv.w)[..., None]).to_array().real               # [..., 3]
    heading = heading / np.linalg.norm(heading, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore"):
        half = np.sqrt(RADIUS**2 - (nearest**2).sum(axis=-1, keepdims=True))
    return np.stack([nearest - half * heading, nearest + half * heading], axis=-2)


def draw_on(ax: plt.Axes, lines: core.Bivector, rulings: core.Bivector, across: core.Bivector, crossing: core.Point) -> None:
    ax.add_collection3d(Line3DCollection(chords(rulings[0]), colors="#9ecae1", linewidths=0.7))
    ax.add_collection3d(Line3DCollection(chords(rulings[1]), colors="#c7c7c7", linewidths=0.7))
    ax.add_collection3d(Line3DCollection(chords(lines[:3]), colors="#08306b", linewidths=2.2))
    ax.add_collection3d(Line3DCollection(chords(lines[3:]), colors="#e67e22", linewidths=2.2))
    visible = real(across.kernel)
    ax.add_collection3d(Line3DCollection(chords(across)[visible], colors="#c0392b", linewidths=2.4))
    points = coordinates(crossing)[real(crossing.kernel)]
    ax.scatter(*points.T, color="#c0392b", s=45, depthshade=False)
    ax.set_title("two real transversals" if visible.all() else "two complex-conjugate transversals", fontsize=11)
    ax.set(xlim=(-RADIUS, RADIUS), ylim=(-RADIUS, RADIUS), zlim=(-RADIUS, RADIUS))
    ax.set_box_aspect((1, 1, 1), zoom=1.7)
    ax.set_axis_off()


def draw(lines: core.Bivector, rulings: core.Bivector, across: core.Bivector, crossing: core.Point,
         azimuth: float) -> plt.Figure:
    """The four lines, the hyperboloid of the first three as its two families of lines, and the
    transversals through the points where the fourth line crosses it."""
    figure = plt.figure(figsize=(6.5, 6.5))
    ax = figure.add_subplot(projection="3d")
    draw_on(ax, lines, rulings, across, crossing)
    ax.view_init(elev=18, azim=azimuth)
    return figure


def animate(scenes: list[tuple[core.Bivector, core.Bivector, core.Bivector, core.Point]]) -> list[np.ndarray]:
    """One frame per scene, the view turning once around the hyperboloid over the loop."""
    frames = []
    for index, scene in enumerate(scenes):
        figure = draw(*scene, azimuth=-60.0 + 360.0 * index / len(scenes))
        frames.append(capture(figure))
        plt.close(figure)
    return frames
