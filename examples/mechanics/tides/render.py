"""Density, a cluster of stars, and the tidal map's prediction of its shape."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm
from matplotlib.contour import ContourSet

from numga import stack
from examples.util.animation import capture
from examples.mechanics.tides import core

STARS = "#0f172a"
PREDICTION = "#ea580c"
ZOOM = 1.6
RIM_PIXELS = 160
# The faintest density drawn, as a fraction of the peak.
DENSITY_FLOOR = 1e-4


# --- plumbing -------------------------------------------------------------------------
def coordinates(points: core.Point) -> np.ndarray:
    """Points in the plane z = 0 `[..., 2]`: their pairings with the planes x and y."""
    return (stack((core.mv.x, core.mv.y)) & points[..., None]).to_array()


def density(ax: Axes, points: core.Point, mass_density: core.Scalar) -> None:
    """The density on a log scale, as an image over the grid's cells."""
    places = coordinates(points)                                             # [rows, columns, 2]
    # The grid samples cell centres; the image extends half a cell beyond them.
    half_cell = (places[1, 1] - places[0, 0]) / 2
    box = (places[0, 0, 0] - half_cell[0], places[0, -1, 0] + half_cell[0],
           places[0, 0, 1] - half_cell[1], places[-1, 0, 1] + half_cell[1])
    values = mass_density.to_array()
    ax.imshow(values, extent=box, origin="lower", cmap="Blues", norm=LogNorm(values.max() * DENSITY_FLOOR, values.max()),
              interpolation="bilinear")
    ax.set(xlim=box[:2], ylim=box[2:], xticks=[], yticks=[])
    ax.set_aspect("equal")


def rim_at(ax: Axes, centre: core.Point, rim: core.Form, half_size: core.Scalar) -> ContourSet:
    """The rim as the level set of its form, evaluated on a grid of offsets around the centre: its contour, for an
    animation to replace."""
    samples = half_size * np.linspace(-1, 1, RIM_PIXELS)
    offsets = core.mv.x.dual() * samples[None, :] + core.mv.y.dual() * samples[:, None]
    level = rim(offsets, offsets).to_array()
    xy = coordinates(centre + offsets)                                       # [rows, columns, 2]
    return ax.contour(xy[..., 0], xy[..., 1], level, levels=[1.0], colors=[PREDICTION], linewidths=1.4)


def frame(points: core.Point, mass_density: core.Scalar, stars: core.Stars, centre: core.Point, rim: core.Form) -> np.ndarray:
    """The stars over the density, beside a close view of the cluster with its predicted rim."""
    return next(animate(points, mass_density, [(stars, centre, rim)]))


def animate(points: core.Point, mass_density: core.Scalar,
            states: Iterable[tuple[core.Stars, core.Point, core.Form]]) -> Iterator[np.ndarray]:
    """One frame per state of the stars, their centre and their predicted rim: the stars over the density, drawn
    once, beside a close view of the cluster with its predicted rim."""
    figure = plt.figure(figsize=(11, 4.4), dpi=80, layout="constrained")
    whole, near = figure.subplots(1, 2, gridspec_kw={"width_ratios": (1.5, 1)})
    density(whole, points, mass_density)
    wide, close = (ax.scatter([], [], s=size, color=STARS, linewidths=0) for ax, size in ((whole, 2), (near, 5)))
    near.set(xticks=[], yticks=[])
    near.set_aspect("equal")
    # The rim drawn for the frame before, which each frame replaces.
    rims = []
    for stars, centre, rim in states:
        positions = coordinates(stars.positions)                             # [stars, 2]
        wide.set_offsets(positions)
        close.set_offsets(positions)
        # Around the cluster, scaled to its extent: the stars' largest offset from the centre across or up.
        middle = coordinates(centre)                                         # [2]
        spans = (stack((core.mv.x, core.mv.y)) & (stars.positions - centre)[..., None]).abs().reshape(-1)
        half_size = spans.max() * ZOOM
        while rims:
            rims.pop().remove()
        rims.append(rim_at(near, centre, rim, half_size))
        frame_size = float(half_size.to_array())
        near.set(xlim=(middle[0] - frame_size, middle[0] + frame_size),
                 ylim=(middle[1] - frame_size, middle[1] + frame_size))
        yield capture(figure)
    plt.close(figure)
