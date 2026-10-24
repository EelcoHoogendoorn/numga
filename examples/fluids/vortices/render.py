"""Vorticity and swirl around moving vortices."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm, to_rgb
from matplotlib.image import AxesImage

from numga import stack

from examples.util.animation import capture
from examples.fluids.vortices import core
from examples.util.wgsl import fitted, show, solids, strokes

# Worldlines: one colour per vortex, the strands' width in pixels, the view as elevation and azimuth in radians,
# azimuth zero looking along +y, the margin around a braid as a share of its larger side, and the size in inches of
# each braid's square panel.
STRANDS = np.array([to_rgb(colour) for colour in ("#c84b31", "#2d6a9f", "#4c956c", "#e0a458", "#8e7dbe")])
STRAND_WIDTH = 3.1
ELEVATION, AZIMUTH = np.radians([18.0, 30.0])
MARGIN = 0.1
PANEL_SIZE = 6.0
BACKGROUND = np.ones(3)


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
           vorticity_limit: float, swirl_limit: float) -> list[AxesImage]:
    vorticity = derivative.cast(core.ga.subspace("xy")).kernel[..., 0]
    left, right = figure.subplots(1, 2)
    return [field(left, points, vorticity, vorticity_limit, "RdBu_r"),
            field(right, points, swirl.kernel[..., 0], swirl_limit, "PuOr_r")]


def figure(points: core.Point, derivative: core.Even, swirl: core.Scalar, vorticity_limit: float, swirl_limit: float) -> plt.Figure:
    result = plt.figure(figsize=(12, 12 / (2 * aspect(points)) + 0.3), layout="constrained")
    panels(result, points, derivative, swirl, vorticity_limit, swirl_limit)
    return result


def animate(frames: Iterable[tuple[core.Vortices, core.Point, core.Even, core.Scalar]],
            vorticity_limit: float, swirl_limit: float) -> Iterator[np.ndarray]:
    """Vorticity and swirl on the window that follows the vortices, frame by frame: the figure set up on the first
    frame, each frame's fields and window set in it."""
    frames = iter(frames)
    first = _, points, derivative, swirl = next(frames)
    result = plt.figure(figsize=(10, 10 / (2 * aspect(points)) + 0.3), dpi=80, layout="constrained")
    images = panels(result, points, derivative, swirl, vorticity_limit, swirl_limit)
    for _, points, derivative, swirl in chain([first], frames):
        box = extent(points)
        for image, values in zip(images, (derivative.cast(core.ga.subspace("xy")).kernel[..., 0], swirl.kernel[..., 0])):
            image.set_data(values)
            image.set_extent(box)
            image.axes.set(xlim=box[:2], ylim=box[2:])
        yield capture(result)
    plt.close(result)


def window(points: core.Point, frames: Iterable[tuple[core.Vortices, core.Even]], vorticity_limit: float) -> Iterator[np.ndarray]:
    """Vorticity on one fixed strip of the plane, frame by frame, each vortex trailing the path it has
    taken. One figure is updated in place and drawn once per frame, without supersampling, to keep
    pace."""
    result = plt.figure(figsize=(12, 12 / aspect(points) + 0.2), dpi=80, layout="constrained")
    ax = result.subplots()
    image = field(ax, points, np.zeros(points.shape), vorticity_limit, "RdBu_r")
    path = ax.add_collection(LineCollection([], colors="#0f172a", linewidths=0.8, alpha=0.5))
    travelled = []
    for vortices, derivative in frames:
        travelled.append(positions(vortices.centres))                          # [vortices, 2] each
        image.set_data(derivative.cast(core.ga.subspace("xy")).kernel[..., 0])
        path.set_segments(np.stack(travelled, axis=1))
        result.canvas.draw()
        yield np.asarray(result.canvas.buffer_rgba())[..., :3].copy()
    plt.close(result)


def braids(worldlines: list[core.Point], height: float) -> plt.Figure:
    """Worldlines `[times, vortices]` side by side, each a braid of strokes through the plane and time, one
    colour per vortex, drawn following the vortices' mean position. Time runs upward and is stretched to
    `height` times the braid's width: it has no scale of its own against distance."""
    result = plt.figure(figsize=(PANEL_SIZE * len(worldlines), PANEL_SIZE))
    for ax, centres in zip(result.subplots(1, len(worldlines)), worldlines):
        following = centres - centres.mean(axis=-1)[..., None]                  # [times, vortices] Point
        places = positions(following)                                            # [times, vortices, 2]
        width = np.ptp(places.reshape(-1, 2), axis=0).max()
        times = np.broadcast_to(np.linspace(0, height * width, len(places))[:, None, None], places.shape[:2] + (1,))
        drawn = solids.mv.point(np.concatenate([places, times], axis=-1))        # [times, vortices] Point
        view = fitted(solids.view([drawn], 0.0, ELEVATION, AZIMUTH, MARGIN, 1), ax)
        # Each vortex's strand, its points through time and its colour.
        strands = [(drawn[:, vortex], colour) for vortex, colour in zip(range(drawn.shape[1]), STRANDS)]
        lines = strokes.polylines(strands, STRAND_WIDTH, view)
        show(ax, solids.draw([], view, BACKGROUND, meshes=[(lines, 1.0)]), view)
    return result
