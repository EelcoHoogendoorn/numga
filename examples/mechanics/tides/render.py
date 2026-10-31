"""Density, a cluster of stars, and the tidal map's prediction of its shape."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm, to_rgba

from numga import stack
from examples.util.animation import capture
from examples.util.wgsl import contours, pinned, solids
from examples.util.wgsl.flow import axes_view
from examples.mechanics.tides import core

STARS = "#0f172a"
PREDICTION = "#ea580c"
ZOOM = 1.6
RIM_PIXELS = 160
# The width of the predicted rim, in pixels.
RIM_WIDTH = 1.6
# The faintest density drawn, as a fraction of the peak.
DENSITY_FLOOR = 1e-4


# --- plumbing -------------------------------------------------------------------------
def coordinates(points: core.Point) -> np.ndarray:
    """Points in the plane z = 0 `[..., 2]`: their pairings with the planes x and y."""
    return (stack((core.mv.x, core.mv.y)) & points[..., None]).to_array()


def density(ax: Axes, points: core.Point, mass_density: core.Scalar) -> None:
    """The density on a log scale over the grid of its samples `[rows, columns]`, from the first to the last."""
    corners = coordinates(points[[0, -1], [0, -1]])                          # [2, 2]
    values = mass_density.to_array()                                         # [rows, columns]
    colours = plt.get_cmap("Blues")(LogNorm(values.max() * DENSITY_FLOOR, values.max())(values))[..., :3]
    ax.set(xlim=corners[:, 0], ylim=corners[:, 1], xticks=[], yticks=[])
    ax.set_aspect("equal")
    places = solids.origin + solids.context.include_from_algebra(points.reshape(-1))   # [rows * columns] Point-
    contours.overlay(ax, places, contours.Mesh.grid(*values.shape), values.reshape(-1), colours.reshape(-1, 3))
    # The panel keeps its frame around the picture.
    ax.set_axis_on()


def rim_at(centre: core.Point, rim: core.Form, half_size: core.Scalar, grid: contours.Mesh,
           view: solids.View) -> np.ndarray:
    """The rim as the level set of its form, evaluated on a grid of offsets around the centre, `RIM_PIXELS` a side
    and numbered row by row, and seen by the view of its panel, white elsewhere: `[rows, columns, 4]`."""
    samples = half_size * np.linspace(-1, 1, RIM_PIXELS)
    offsets = core.mv.x.dual() * samples[None, :] + core.mv.y.dual() * samples[:, None]   # [rim, rim] Point-
    level = rim(offsets, offsets).to_array()                                 # [rim, rim]
    places = solids.origin + solids.context.include_from_algebra((centre + offsets).reshape(-1))   # [rim * rim] Point-
    prediction = contours.Level(1.0, np.array(to_rgba(PREDICTION)), RIM_WIDTH)
    return contours.picture(places, grid, level.reshape(-1), np.ones(3), view, levels=(prediction,))


def frame(points: core.Point, mass_density: core.Scalar, stars: core.Stars, centre: core.Point, rim: core.Form) -> np.ndarray:
    """The stars over the density, beside a close view of the cluster with its predicted rim."""
    return next(animate(points, mass_density, [(stars, centre, rim)]))


def animate(points: core.Point, mass_density: core.Scalar,
            states: Iterable[tuple[core.Stars, core.Point, core.Form]]) -> Iterator[np.ndarray]:
    """One frame per state of the stars, their centre and their predicted rim: the stars over the density, drawn
    once, beside a close view of the cluster with its predicted rim."""
    figure = plt.figure(figsize=(11, 4.4), dpi=80, layout="constrained")
    whole, near = figure.subplots(1, 2, gridspec_kw={"width_ratios": (1.5, 1)})
    # Both panels bare of ticks before either picture is fitted to the room the layout gives it.
    near.set(xlim=(-1, 1), ylim=(-1, 1), xticks=[], yticks=[])
    near.set_aspect("equal")
    density(whole, points, mass_density)
    wide, close = (ax.scatter([], [], s=size, color=STARS, linewidths=0) for ax, size in ((whole, 2), (near, 5)))
    # The rim's picture pinned once, its data and extent set for each frame; the panel keeps its frame around it.
    rims = pinned(near, np.zeros(axes_view(near).shape + (4,)), (-1, 1, -1, 1))
    near.set_axis_on()
    rim_grid = contours.Mesh.grid(RIM_PIXELS, RIM_PIXELS)
    for stars, centre, rim in states:
        positions = coordinates(stars.positions)                             # [stars, 2]
        wide.set_offsets(positions)
        close.set_offsets(positions)
        # Around the cluster, scaled to its extent: the stars' largest offset from the centre across or up.
        middle = coordinates(centre)                                         # [2]
        spans = (stack((core.mv.x, core.mv.y)) & (stars.positions - centre)[..., None]).abs().reshape(-1)
        half_size = spans.max() * ZOOM
        frame_size = float(half_size.to_array())
        window = (middle[0] - frame_size, middle[0] + frame_size, middle[1] - frame_size, middle[1] + frame_size)
        rims.set_extent(window)
        near.set(xlim=window[:2], ylim=window[2:])
        rims.set_data(rim_at(centre, rim, half_size, rim_grid, axes_view(near)))
        yield capture(figure)
    plt.close(figure)
