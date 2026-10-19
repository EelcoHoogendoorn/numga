"""Pressure, streamlines and lift around a wing."""

from __future__ import annotations

from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from numga import stack

from examples.animation import capture
from examples.mechanics.wing import core

WINDOW = (-3.2, 3.2, -1.8, 1.8)
PRESSURE_RANGE = (-3.0, 1.0)
STREAM_LEVELS = np.linspace(-3.0, 3.0, 21)
LIFT_SCALE = 0.35
WING = "#0f172a"
LIFT = "#16a34a"
MARK = "#facc15"
CENTRE = "#f8fafc"
# The animations' dots per inch.
FRAME_DPI = 36


# --- plumbing -------------------------------------------------------------------------
def coordinates(points: core.Point) -> np.ndarray:
    """x and y of each point, on the last axis: its pairings with the lines x and y."""
    return (stack((core.mv.x, core.mv.y)) & points[..., None]).to_array()


def draw(ax, points: core.Point, velocity: core.Point, stream: core.Bivector, lift: core.Point, speed: float) -> None:
    """The pressure coefficient around the wing at the points, the streamlines, the wing, and its lift."""
    # Each ring closed, its first angle repeated at the end, so no seam is left unpainted.
    def closed(values: np.ndarray) -> np.ndarray:
        return np.concatenate([values, values[:, :1]], axis=1)

    xy = closed(coordinates(points))
    speed_squared = closed((velocity | velocity).kernel[..., 0])
    # The pressure coefficient: low where the flow is fast.
    pressure = 1 - speed_squared / speed**2
    ax.pcolormesh(xy[..., 0], xy[..., 1], pressure, cmap="RdBu", shading="gouraud",
                  norm=TwoSlopeNorm(0.0, *PRESSURE_RANGE))
    ax.contour(xy[..., 0], xy[..., 1], closed(stream.cast(core.ga.subspace("xy")).kernel[..., 0]), levels=STREAM_LEVELS,
               colors="#334155", linewidths=0.6)
    ax.fill(xy[0, :, 0], xy[0, :, 1], color=WING)
    arrow = coordinates(lift) * LIFT_SCALE
    ax.annotate("", xy=arrow, xytext=(0.0, 0.0), arrowprops=dict(arrowstyle="-|>", color=LIFT, linewidth=2.5))
    ax.set(xlim=WINDOW[:2], ylim=WINDOW[2:], xticks=[], yticks=[])
    ax.set_aspect("equal")


def figure(points: core.Point, velocity: core.Point, stream: core.Bivector, lift: core.Point, speed: float) -> plt.Figure:
    result = plt.figure(figsize=(8, 4.6), layout="constrained")
    draw(result.subplots(), points, velocity, stream, lift, speed)
    return result


def frame(points: core.Point, velocity: core.Point, stream: core.Bivector, lift: core.Point, speed: float,
          critical: core.Point, centre: core.Point) -> np.ndarray:
    """One frame, with the images of the map's critical points and the cylinder's centre marked."""
    result = plt.figure(figsize=(8, 4.6), dpi=FRAME_DPI, layout="constrained")
    ax = result.subplots()
    draw(ax, points, velocity, stream, lift, speed)
    for marks, colour in ((critical, MARK), (centre, CENTRE)):
        xy = coordinates(marks)
        ax.scatter(xy[..., 0], xy[..., 1], s=30, color=colour, edgecolors=WING, linewidths=0.8, zorder=3)
    pixels = capture(result)
    plt.close(result)
    return pixels


def animate_marked(frames: Iterable[tuple[core.Point, core.Point, core.Bivector, core.Point, core.Point, core.Point]],
                   speed: float) -> list[np.ndarray]:
    """Each frame of points, velocity, stream function and lift, with the critical points' images and the
    cylinder's centre marked."""
    return [frame(points, velocity, stream, lift, speed, critical, centre)
            for points, velocity, stream, lift, critical, centre in frames]


def animate(frames: Iterable[tuple[core.Point, core.Point, core.Bivector, core.Point]], speed: float) -> list[np.ndarray]:
    """Each frame of points, velocity, stream function and lift."""
    return [plain_frame(points, velocity, stream, lift, speed) for points, velocity, stream, lift in frames]


def plain_frame(points: core.Point, velocity: core.Point, stream: core.Bivector, lift: core.Point, speed: float) -> np.ndarray:
    result = plt.figure(figsize=(8, 4.6), dpi=FRAME_DPI, layout="constrained")
    draw(result.subplots(), points, velocity, stream, lift, speed)
    pixels = capture(result)
    plt.close(result)
    return pixels


def bipolar(grid: core.Point, grid_images: core.Point, circle: core.Point, circle_image: core.Point,
            critical: core.Point) -> plt.Figure:
    """The grid of circles through and around the critical points with the wing's circle, before and
    after the map, with the critical points and where they move."""
    result = plt.figure(figsize=(10, 4.2), layout="constrained")
    for ax, curves, outline, marks in zip(
            result.subplots(1, 2), (grid, grid_images), (circle, circle_image), (critical, 2 * critical)):
        for curve in coordinates(curves):
            ax.plot(curve[:, 0], curve[:, 1], color="#94a3b8", linewidth=0.9)
        xy = coordinates(outline)
        ax.plot(xy[:, 0], xy[:, 1], color=WING, linewidth=2.2)
        ends = coordinates(marks)
        ax.scatter([ends[0], -ends[0]], [ends[1], -ends[1]], color=MARK, edgecolors=WING, zorder=4)
        ax.set(xlim=WINDOW[:2], ylim=WINDOW[2:], xticks=[], yticks=[])
        ax.set_aspect("equal")
    return result
