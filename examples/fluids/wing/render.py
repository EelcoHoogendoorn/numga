"""Pressure, streamlines and lift around a wing."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from numga import Extensor, concatenate, stack

from examples.util.animation import capture
from examples.fluids.wing import core

window = (-3.2, 3.2, -1.8, 1.8)
pressure_range = (-3.0, 1.0)
stream_levels = np.linspace(-3.0, 3.0, 21)
lift_scale = 0.35
wing_colour = "#0f172a"
lift_colour = "#16a34a"
mark_colour = "#facc15"
centre_colour = "#f8fafc"
# The animations' dots per inch: the README's wing.gif 400 pixels across.
frame_dpi = 50


# --- plumbing -------------------------------------------------------------------------
def coordinates(points: core.Point) -> np.ndarray:
    """x and y of each point, on the last axis: its pairings with the lines x and y."""
    return (stack((core.mv.x, core.mv.y)) & points[..., None]).to_array()


def components(force: core.Force) -> np.ndarray:
    """A force's x and y components, its pairings with the unit displacements along x and y."""
    return (force & stack((core.mv.x, core.mv.y)).dual()).to_array()


def draw(ax, points: core.Point, velocity: core.Point, stream: core.Bivector, lift: core.Force, speed: float) -> None:
    """The pressure coefficient around the wing at the points, the streamlines, the wing, and its lift."""
    # Each ring closed, its first angle repeated at the end, so no seam is left unpainted.
    def closed(values: Extensor) -> Extensor:
        return concatenate([values, values[:, :1]], axis=1)

    xy = coordinates(closed(points))
    speed_squared = closed(velocity.dot(velocity)).to_array()
    # The pressure coefficient: low where the flow is fast.
    pressure = 1 - speed_squared / speed**2
    ax.pcolormesh(xy[..., 0], xy[..., 1], pressure, cmap="RdBu", shading="gouraud",
                  norm=TwoSlopeNorm(0.0, *pressure_range))
    # The stream function's values, its pairings with the unit scalar.
    stream_values = closed(stream & core.mv.scalar([1])).to_array()
    ax.contour(xy[..., 0], xy[..., 1], stream_values, levels=stream_levels, colors="#334155", linewidths=0.6)
    ax.fill(xy[0, :, 0], xy[0, :, 1], color=wing_colour)
    arrow = components(lift * lift_scale)
    ax.annotate("", xy=arrow, xytext=(0.0, 0.0), arrowprops=dict(arrowstyle="-|>", color=lift_colour, linewidth=2.5))
    ax.set(xlim=window[:2], ylim=window[2:], xticks=[], yticks=[])
    ax.set_aspect("equal")


def figure(points: core.Point, velocity: core.Point, stream: core.Bivector, lift: core.Force, speed: float) -> plt.Figure:
    result = plt.figure(figsize=(8, 4.6), dpi=frame_dpi, layout="constrained")
    draw(result.subplots(), points, velocity, stream, lift, speed)
    return result


def frame(points: core.Point, velocity: core.Point, stream: core.Bivector, lift: core.Force, speed: float,
          critical: core.Point, centre: core.Point) -> np.ndarray:
    """One frame, with the images of the map's critical points and the cylinder's centre marked."""
    result = plt.figure(figsize=(8, 4.6), dpi=frame_dpi, layout="constrained")
    ax = result.subplots()
    draw(ax, points, velocity, stream, lift, speed)
    for marks, colour in ((critical, mark_colour), (centre, centre_colour)):
        xy = coordinates(marks)
        ax.scatter(xy[..., 0], xy[..., 1], s=30, color=colour, edgecolors=wing_colour, linewidths=0.8, zorder=3)
    pixels = capture(result)
    plt.close(result)
    return pixels


def animate_marked(frames: Iterable[tuple[core.Point, core.Point, core.Bivector, core.Force, core.Point, core.Point]],
                   speed: float) -> Iterator[np.ndarray]:
    """Each frame of points, velocity, stream function and lift, with the critical points' images and the
    cylinder's centre marked."""
    for points, velocity, stream, lift, critical, centre in frames:
        yield frame(points, velocity, stream, lift, speed, critical, centre)


# README picture: this makes plots/wing.gif, tuned as a visual for the README. Don't change it; to
# experiment, copy it.
def animate(frames: Iterable[tuple[core.Point, core.Point, core.Bivector, core.Force]], speed: float) -> Iterator[np.ndarray]:
    """Each frame of points, velocity, stream function and lift."""
    for points, velocity, stream, lift in frames:
        yield plain_frame(points, velocity, stream, lift, speed)


def plain_frame(points: core.Point, velocity: core.Point, stream: core.Bivector, lift: core.Force, speed: float) -> np.ndarray:
    result = plt.figure(figsize=(8, 4.6), dpi=frame_dpi, layout="constrained")
    draw(result.subplots(), points, velocity, stream, lift, speed)
    pixels = capture(result)
    plt.close(result)
    return pixels


def bipolar(grid: core.Point, grid_images: core.Point, circle: core.Point, circle_image: core.Point,
            critical: core.Point) -> plt.Figure:
    """The grid of circles through and around the critical points with the wing's circle, before and
    after the map, with the critical points and where they move."""
    result = plt.figure(figsize=(10, 4.2), dpi=40, layout="constrained")
    for ax, curves, outline, marks in zip(
            result.subplots(1, 2), (grid, grid_images), (circle, circle_image), (critical, 2 * critical)):
        for curve in coordinates(curves):
            ax.plot(curve[:, 0], curve[:, 1], color="#94a3b8", linewidth=0.9)
        xy = coordinates(outline)
        ax.plot(xy[:, 0], xy[:, 1], color=wing_colour, linewidth=2.2)
        # The marks and their reflections through the origin.
        ends = coordinates(stack([marks, -marks]))
        ax.scatter(ends[:, 0], ends[:, 1], color=mark_colour, edgecolors=wing_colour, zorder=4)
        ax.set(xlim=window[:2], ylim=window[2:], xticks=[], yticks=[])
        ax.set_aspect("equal")
    return result
