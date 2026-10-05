"""Vorticity and swirl around moving vortices."""

from __future__ import annotations

from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm
from matplotlib.image import AxesImage

from examples.animation import capture
from examples.mechanics.vortices import core


# --- plumbing -------------------------------------------------------------------------
def extent(points: core.Vector) -> tuple[float, float, float, float]:
    """The box the grid's cells cover: its samples sit at the cell centres, half a cell inside."""
    coordinates = points.cast(core.ga.subspace("x y")).kernel
    half_cell = (coordinates[1, 1] - coordinates[0, 0]) / 2
    low, high = coordinates[0, 0] - half_cell, coordinates[-1, -1] + half_cell
    return low[0], high[0], low[1], high[1]


def aspect(points: core.Vector) -> float:
    left, right, bottom, top = extent(points)
    return (right - left) / (top - bottom)


def field(ax: Axes, points: core.Vector, values: np.ndarray, limit: float, colours: str) -> AxesImage:
    box = extent(points)
    image = ax.imshow(values, extent=box, origin="lower", cmap=colours, interpolation="bilinear",
                      norm=TwoSlopeNorm(vcenter=0.0, vmin=-limit, vmax=limit))
    ax.set(xlim=box[:2], ylim=box[2:], xticks=[], yticks=[])
    ax.autoscale(False)
    ax.set_aspect("equal")
    return image


def panels(figure: plt.Figure, points: core.Vector, derivative: core.Even, swirl: core.Scalar,
           vorticity_limit: float, swirl_limit: float) -> None:
    vorticity = derivative.cast(core.ga.subspace("xy")).kernel[..., 0]
    left, right = figure.subplots(1, 2)
    field(left, points, vorticity, vorticity_limit, "RdBu_r")
    field(right, points, swirl.kernel[..., 0], swirl_limit, "PuOr_r")


def figure(points: core.Vector, derivative: core.Even, swirl: core.Scalar, vorticity_limit: float, swirl_limit: float) -> plt.Figure:
    result = plt.figure(figsize=(12, 12 / (2 * aspect(points)) + 0.3), layout="constrained")
    panels(result, points, derivative, swirl, vorticity_limit, swirl_limit)
    return result


def frame(points: core.Vector, derivative: core.Even, swirl: core.Scalar, vorticity_limit: float, swirl_limit: float) -> np.ndarray:
    result = plt.figure(figsize=(10, 10 / (2 * aspect(points)) + 0.3), dpi=80, layout="constrained")
    panels(result, points, derivative, swirl, vorticity_limit, swirl_limit)
    pixels = capture(result)
    plt.close(result)
    return pixels


def animate(frames: Iterable[tuple[core.Vortices, core.Vector, core.Even, core.Scalar]],
            vorticity_limit: float, swirl_limit: float) -> list[np.ndarray]:
    return [frame(points, derivative, swirl, vorticity_limit, swirl_limit) for _, points, derivative, swirl in frames]


def window(points: core.Vector, frames: Iterable[tuple[core.Vortices, core.Even]], vorticity_limit: float) -> list[np.ndarray]:
    """Vorticity on one fixed strip of the plane, frame by frame, each vortex trailing the path it has
    taken. One figure is updated in place and drawn once per frame, without supersampling, to keep
    pace."""
    result = plt.figure(figsize=(10, 10 / aspect(points) + 0.2), dpi=80, layout="constrained")
    ax = result.subplots()
    image = field(ax, points, np.zeros(points.shape), vorticity_limit, "RdBu_r")
    path = ax.add_collection(LineCollection([], colors="#0f172a", linewidths=0.8, alpha=0.5))
    travelled = []
    pixels = []
    for vortices, derivative in frames:
        travelled.append(vortices.centres.cast(core.ga.subspace("x y")).kernel)    # [vortices, 2] each
        image.set_data(derivative.cast(core.ga.subspace("xy")).kernel[..., 0])
        path.set_segments(np.stack(travelled, axis=1))
        result.canvas.draw()
        pixels.append(np.asarray(result.canvas.buffer_rgba())[..., :3].copy())
    plt.close(result)
    return pixels


def inline(frames: list[np.ndarray], duration_ms: int):
    """Frames as a looping GIF to show in a notebook, kept in memory."""
    from io import BytesIO
    from IPython.display import Image as Shown
    from PIL import Image
    images = [Image.fromarray(pixels) for pixels in frames]
    buffer = BytesIO()
    images[0].save(buffer, format="GIF", save_all=True, append_images=images[1:], duration=duration_ms, loop=0)
    return Shown(data=buffer.getvalue(), format="gif")
