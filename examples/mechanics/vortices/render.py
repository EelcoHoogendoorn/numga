"""Vorticity and swirl around moving vortices."""

from __future__ import annotations

from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import TwoSlopeNorm

from examples.animation import capture
from examples.mechanics.vortices import core

TURNING = "#ea580c"
COUNTER = "#2563eb"


# --- plumbing -------------------------------------------------------------------------
def field(ax: Axes, points: core.Vector, values: np.ndarray, limit: float, colours: str, title: str) -> None:
    coordinates = points.cast(core.ga.subspace("x y")).kernel
    box = (coordinates[0, 0, 0], coordinates[0, -1, 0], coordinates[0, 0, 1], coordinates[-1, 0, 1])
    ax.imshow(values, extent=box, origin="lower", cmap=colours, interpolation="bilinear",
              norm=TwoSlopeNorm(vcenter=0.0, vmin=-limit, vmax=limit))
    ax.set_title(title, fontsize=11)
    ax.set(xlim=box[:2], ylim=box[2:])
    ax.autoscale(False)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def centres(ax: Axes, vortices: core.Vortices) -> None:
    positions = (vortices.centres[..., None] + vortices.copies).cast(core.ga.subspace("x y")).kernel
    colours = np.repeat(np.where(vortices.circulations > 0, TURNING, COUNTER), positions.shape[1])
    ax.scatter(positions[..., 0].ravel(), positions[..., 1].ravel(), s=10, c=colours,
               edgecolors="white", linewidths=0.6, zorder=3)


def panels(figure: plt.Figure, vortices: core.Vortices, points: core.Vector, derivative: core.Even, swirl: core.Scalar,
           vorticity_limit: float, swirl_limit: float) -> None:
    vorticity = derivative.cast(core.ga.subspace("xy")).kernel[..., 0]
    left, right = figure.subplots(1, 2)
    field(left, points, vorticity, vorticity_limit, "RdBu_r", "Vorticity, the bivector of the derivative")
    field(right, points, swirl.kernel[..., 0], swirl_limit, "PuOr_r", "Swirl against strain")
    for ax in (left, right):
        centres(ax, vortices)


def figure(vortices: core.Vortices, points: core.Vector, derivative: core.Even, swirl: core.Scalar,
           vorticity_limit: float, swirl_limit: float) -> plt.Figure:
    result = plt.figure(figsize=(12, 4.4), layout="constrained")
    panels(result, vortices, points, derivative, swirl, vorticity_limit, swirl_limit)
    return result


def frame(vortices: core.Vortices, points: core.Vector, derivative: core.Even, swirl: core.Scalar,
          vorticity_limit: float, swirl_limit: float) -> np.ndarray:
    coordinates = points.cast(core.ga.subspace("x y")).kernel
    aspect = (coordinates[0, -1, 0] - coordinates[0, 0, 0]) / (coordinates[-1, 0, 1] - coordinates[0, 0, 1])
    result = plt.figure(figsize=(10, 10 / (2 * aspect) + 0.5), dpi=80, layout="constrained")
    panels(result, vortices, points, derivative, swirl, vorticity_limit, swirl_limit)
    pixels = capture(result)
    plt.close(result)
    return pixels


def animate(frames: Iterable[tuple[core.Vortices, core.Vector, core.Even, core.Scalar]],
            vorticity_limit: float, swirl_limit: float) -> list[np.ndarray]:
    return [frame(*state, vorticity_limit, swirl_limit) for state in frames]
