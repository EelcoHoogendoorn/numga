"""Vorticity and swirl around moving vortices."""

from __future__ import annotations

from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm
from matplotlib.image import AxesImage
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from numga import stack

from examples.animation import capture
from examples.mechanics.vortices import core

# Worldlines: one colour per vortex, the strands' width, the view as (elevation, azimuth) in degrees,
# and the size in inches of each braid's square panel.
STRANDS = np.array(["#c84b31", "#2d6a9f", "#4c956c", "#e0a458", "#8e7dbe"])
STRAND_WIDTH = 2.0
ELEVATION, AZIMUTH = 18.0, -60.0
PANEL_SIZE = 6.0


# --- plumbing -------------------------------------------------------------------------
def positions(points: core.Point) -> np.ndarray:
    """x and y of each point, on the last axis: its pairings with the lines x and y."""
    return (stack((core.mv.x, core.mv.y)) & points[..., None]).to_array()


def extent(points: core.Point) -> tuple[float, float, float, float]:
    """The box the grid's cells cover: its samples sit at the cell centres, half a cell inside."""
    coordinates = positions(points)
    half_cell = (coordinates[1, 1] - coordinates[0, 0]) / 2
    low, high = coordinates[0, 0] - half_cell, coordinates[-1, -1] + half_cell
    return low[0], high[0], low[1], high[1]


def aspect(points: core.Point) -> float:
    left, right, bottom, top = extent(points)
    return (right - left) / (top - bottom)


def field(ax: Axes, points: core.Point, values: np.ndarray, limit: float, colours: str) -> AxesImage:
    box = extent(points)
    image = ax.imshow(values, extent=box, origin="lower", cmap=colours, interpolation="bilinear",
                      norm=TwoSlopeNorm(vcenter=0.0, vmin=-limit, vmax=limit))
    ax.set(xlim=box[:2], ylim=box[2:], xticks=[], yticks=[])
    ax.autoscale(False)
    ax.set_aspect("equal")
    return image


def panels(figure: plt.Figure, points: core.Point, derivative: core.Even, swirl: core.Scalar,
           vorticity_limit: float, swirl_limit: float) -> None:
    vorticity = derivative.cast(core.ga.subspace("xy")).kernel[..., 0]
    left, right = figure.subplots(1, 2)
    field(left, points, vorticity, vorticity_limit, "RdBu_r")
    field(right, points, swirl.kernel[..., 0], swirl_limit, "PuOr_r")


def figure(points: core.Point, derivative: core.Even, swirl: core.Scalar, vorticity_limit: float, swirl_limit: float) -> plt.Figure:
    result = plt.figure(figsize=(12, 12 / (2 * aspect(points)) + 0.3), layout="constrained")
    panels(result, points, derivative, swirl, vorticity_limit, swirl_limit)
    return result


def frame(points: core.Point, derivative: core.Even, swirl: core.Scalar, vorticity_limit: float, swirl_limit: float) -> np.ndarray:
    result = plt.figure(figsize=(10, 10 / (2 * aspect(points)) + 0.3), dpi=80, layout="constrained")
    panels(result, points, derivative, swirl, vorticity_limit, swirl_limit)
    pixels = capture(result)
    plt.close(result)
    return pixels


def animate(frames: Iterable[tuple[core.Vortices, core.Point, core.Even, core.Scalar]],
            vorticity_limit: float, swirl_limit: float) -> list[np.ndarray]:
    return [frame(points, derivative, swirl, vorticity_limit, swirl_limit) for _, points, derivative, swirl in frames]


def window(points: core.Point, frames: Iterable[tuple[core.Vortices, core.Even]], vorticity_limit: float) -> list[np.ndarray]:
    """Vorticity on one fixed strip of the plane, frame by frame, each vortex trailing the path it has
    taken. One figure is updated in place and drawn once per frame, without supersampling, to keep
    pace."""
    result = plt.figure(figsize=(12, 12 / aspect(points) + 0.2), dpi=80, layout="constrained")
    ax = result.subplots()
    image = field(ax, points, np.zeros(points.shape), vorticity_limit, "RdBu_r")
    path = ax.add_collection(LineCollection([], colors="#0f172a", linewidths=0.8, alpha=0.5))
    travelled = []
    pixels = []
    for vortices, derivative in frames:
        travelled.append(positions(vortices.centres))                          # [vortices, 2] each
        image.set_data(derivative.cast(core.ga.subspace("xy")).kernel[..., 0])
        path.set_segments(np.stack(travelled, axis=1))
        result.canvas.draw()
        pixels.append(np.asarray(result.canvas.buffer_rgba())[..., :3].copy())
    plt.close(result)
    return pixels


def braids(worldlines: list[core.Point], height: float) -> plt.Figure:
    """Worldlines `[times, vortices]` side by side, each a braid of curves through the plane and time, one
    colour per vortex, drawn following the vortices' mean position. Time runs upward and is stretched to
    `height` times the braid's width: it has no scale of its own against distance."""
    result = plt.figure(figsize=(PANEL_SIZE * len(worldlines), PANEL_SIZE))
    toward = np.array([np.cos(np.radians(ELEVATION)) * np.cos(np.radians(AZIMUTH)),
                       np.cos(np.radians(ELEVATION)) * np.sin(np.radians(AZIMUTH)), np.sin(np.radians(ELEVATION))])
    for ax, centres in zip(result.subplots(1, len(worldlines), subplot_kw={"projection": "3d"}), worldlines):
        following = centres - centres.mean(axis=-1)[..., None]                  # [times, vortices] Point
        places = positions(following)                                            # [times, vortices, 2]
        width = np.ptp(places.reshape(-1, 2), axis=0).max()
        times = np.broadcast_to(np.linspace(0, height * width, len(places))[:, None, None], places.shape[:2] + (1,))
        drawn = np.concatenate([places, times], axis=-1)                         # [times, vortices, 3]
        segments = np.stack([drawn[:-1], drawn[1:]], axis=-2).reshape(-1, 2, 3)  # [(times - 1) * vortices, 2, 3]
        colours = np.broadcast_to(STRANDS[: places.shape[1]], drawn[1:].shape[:2]).reshape(-1)
        # The far segments first, so that a strand passing in front covers the one behind it.
        order = np.argsort(segments.mean(axis=1) @ toward)
        ax.computed_zorder = False
        ax.add_collection3d(Line3DCollection(segments[order], colors=colours[order], linewidths=STRAND_WIDTH, capstyle="round"))
        low, high = drawn.reshape(-1, 3).min(axis=0), drawn.reshape(-1, 3).max(axis=0)
        ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), zlim=(low[2], high[2]))
        ax.set_box_aspect(tuple(high - low))
        ax.set_proj_type("ortho")
        ax.view_init(elev=ELEVATION, azim=AZIMUTH)
        ax.set_axis_off()
    return result
