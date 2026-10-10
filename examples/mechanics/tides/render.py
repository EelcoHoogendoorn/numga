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
# The faintest density drawn, as a fraction of the peak.
DENSITY_FLOOR = 1e-4


# --- plumbing -------------------------------------------------------------------------
def density(ax: Axes, points: core.Vector, mass_density: core.Scalar) -> None:
    """The density on a log scale, as an image over the grid's cells."""
    coordinates = points.cast(core.ga.subspace("x y")).kernel
    # The grid samples cell centres; the image extends half a cell beyond them.
    half_cell = (coordinates[1, 1] - coordinates[0, 0]) / 2
    box = (coordinates[0, 0, 0] - half_cell[0], coordinates[0, -1, 0] + half_cell[0],
           coordinates[0, 0, 1] - half_cell[1], coordinates[-1, 0, 1] + half_cell[1])
    values = mass_density.cast(core.ga.subspace.scalar()).kernel[..., 0]
    ax.imshow(values, extent=box, origin="lower", cmap="Blues", norm=LogNorm(values.max() * DENSITY_FLOOR, values.max()),
              interpolation="bilinear")
    ax.set(xlim=box[:2], ylim=box[2:], xticks=[], yticks=[])
    ax.set_aspect("equal")


def stars_at(ax: Axes, stars: core.Stars, size: float) -> None:
    """The stars as dots of the given size."""
    positions = stars.positions.cast(core.ga.subspace("x y")).kernel
    ax.scatter(positions[:, 0], positions[:, 1], s=size, color=STARS, linewidths=0)


def rim_at(ax: Axes, centre: core.Vector, rim: core.Form, half_size: float) -> None:
    """The rim as the level set of its form, evaluated on a grid of offsets around the centre."""
    offsets = core.grid(half_size, half_size, RIM_PIXELS, RIM_PIXELS)
    level = rim(offsets, offsets).kernel[..., 0]
    xy = (centre + offsets).cast(core.ga.subspace("x y")).kernel
    ax.contour(xy[..., 0], xy[..., 1], level, levels=[1.0], colors=[PREDICTION], linewidths=1.4)


def frame(points: core.Vector, mass_density: core.Scalar, stars: core.Stars, centre: core.Vector, rim: core.Form) -> np.ndarray:
    """The stars over the density, beside a close view of the cluster with its predicted rim."""
    figure = plt.figure(figsize=(11, 4.4), dpi=80, layout="constrained")
    whole, near = figure.subplots(1, 2, gridspec_kw={"width_ratios": (1.5, 1)})
    density(whole, points, mass_density)
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


def animate(points: core.Vector, mass_density: core.Scalar,
            states: Iterable[tuple[core.Stars, core.Vector, core.Form]]) -> list[np.ndarray]:
    """One frame per state of the stars, their centre and their predicted rim."""
    return [frame(points, mass_density, *state) for state in states]
