"""Density, a cluster of stars, and the tidal map's prediction of its shape."""

from __future__ import annotations

from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm

from examples.animation import capture
from examples.mechanics.tides import core

STARS = "#0f172a"
PREDICTION = "#ea580c"
ZOOM = 1.6
RIM_PIXELS = 160


# --- plumbing -------------------------------------------------------------------------
def density(ax: Axes, points: core.Vector, derivative: core.Even) -> None:
    coordinates = points.cast(core.ga.subspace("x y")).kernel
    box = (coordinates[0, 0, 0], coordinates[0, -1, 0], coordinates[0, 0, 1], coordinates[-1, 0, 1])
    values = -derivative.cast(core.ga.subspace.scalar()).kernel[..., 0] / (4 * np.pi)
    ax.imshow(values, extent=box, origin="lower", cmap="Blues", norm=LogNorm(values.max() * 1e-4, values.max()),
              interpolation="bilinear")
    ax.set(xlim=box[:2], ylim=box[2:], xticks=[], yticks=[])
    ax.set_aspect("equal")


def stars_at(ax: Axes, stars: core.Stars, size: float) -> None:
    positions = stars.positions.cast(core.ga.subspace("x y")).kernel
    ax.scatter(positions[:, 0], positions[:, 1], s=size, color=STARS, linewidths=0)


def rim_at(ax: Axes, centre: core.Vector, rim: core.Form, half_size: float) -> None:
    """The rim as the level set of its form, evaluated on a grid of offsets around the centre."""
    offsets = core.grid(half_size, half_size, RIM_PIXELS, RIM_PIXELS)
    level = rim(offsets, offsets).kernel[..., 0]
    xy = (centre + offsets).cast(core.ga.subspace("x y")).kernel
    ax.contour(xy[..., 0], xy[..., 1], level, levels=[1.0], colors=[PREDICTION], linewidths=1.4)


def frame(points: core.Vector, derivative: core.Even, stars: core.Stars, centre: core.Vector, rim: core.Form) -> np.ndarray:
    figure = plt.figure(figsize=(11, 4.4), dpi=80, layout="constrained")
    whole, near = figure.subplots(1, 2, gridspec_kw={"width_ratios": (1.5, 1)})
    density(whole, points, derivative)
    stars_at(whole, stars, 2)
    # Around the cluster, scaled to its extent.
    middle = centre.cast(core.ga.subspace("x y")).kernel
    half_size = np.abs(stars.positions.cast(core.ga.subspace("x y")).kernel - middle).max() * ZOOM
    stars_at(near, stars, 5)
    rim_at(near, centre, rim, half_size)
    near.set(xlim=(middle[0] - half_size, middle[0] + half_size), ylim=(middle[1] - half_size, middle[1] + half_size),
             xticks=[], yticks=[])
    near.set_aspect("equal")
    pixels = capture(figure)
    plt.close(figure)
    return pixels


def animate(points: core.Vector, derivative: core.Even,
            states: Iterable[tuple[core.Stars, core.Vector, core.Form]]) -> list[np.ndarray]:
    return [frame(points, derivative, *state) for state in states]
