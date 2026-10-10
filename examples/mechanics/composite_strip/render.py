"""The laminate midsurfaces under a growing pull, side by side in one scene."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection
from contourpy import contour_generator
from matplotlib.colors import to_rgb
from scipy.ndimage import map_coordinates

from examples.animation import capture
from examples.mechanics.composite_strip import core

# The frames' resolution, set by the strips' detail; the view's turn about the vertical and its
# tilt down from the horizontal; the strip's colour and the light it is shaded by.
DPI = 80
AZIMUTH, ELEVATION = np.deg2rad(-74.0), np.deg2rad(22.0)
FACE = np.array(to_rgb("#c6a15b"))
LIGHT = np.array([0.2, -0.75, 0.6]) / np.linalg.norm([0.2, -0.75, 0.6])
# The drawn mesh: every few samples along the length and across the width.
MESH = (slice(None, None, 5), slice(None, None, 2))


# --- plumbing -------------------------------------------------------------------------
def projected(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Points `[..., 3]` in an orthographic view: their screen positions `[..., 2]` and depth `[...]`."""
    x, y, z = np.moveaxis(points, -1, 0)
    across = x * np.cos(AZIMUTH) + y * np.sin(AZIMUTH)
    away = -x * np.sin(AZIMUTH) + y * np.cos(AZIMUTH)
    return (np.stack([across, z * np.cos(ELEVATION) + away * np.sin(ELEVATION)], axis=-1),
            away * np.cos(ELEVATION) - z * np.sin(ELEVATION))


def level_lines(field: np.ndarray, spacing: float) -> list[np.ndarray]:
    """The level lines of a sampled field `[rows, columns]` at every multiple of `spacing` in its range,
    as `[vertices, 2]` arrays of fractional (row, column) sample positions."""
    generator = contour_generator(z=field, line_type="Separate")
    levels = spacing * np.arange(np.ceil(field.min() / spacing), np.floor(field.max() / spacing) + 1)
    return [line[:, ::-1] for level in levels for line in generator.lines(level)]


def animate_strips(rest: core.Vector, frames: core.Vector, across_fibres: core.Scalar, spacing: float) -> list[np.ndarray]:
    """The strips deformed at each frame, `[frames, cases, length samples, width samples]`, seen from
    behind the ends that stay in place: shaded quadrilaterals of a coarse mesh, drawn far to near, with
    the top ply's fibres over them, the level lines of `across_fibres`, `[cases, length samples, width
    samples]`, `spacing` apart, carried on each deformed surface. `rest` sets the common framing."""
    rests = rest.cast(core.ga.subspace("x y z")).kernel
    deformed = frames.cast(core.ga.subspace("x y z")).kernel           # [frames, cases, length samples, width samples, 3]
    surfaces = deformed[(Ellipsis, *MESH, slice(None))]
    screen, _ = projected(np.concatenate((rests[None][(Ellipsis, *MESH, slice(None))], surfaces)))
    low, high = screen.reshape(-1, 2).min(axis=0), screen.reshape(-1, 2).max(axis=0)
    margin = 0.04 * (high - low).max()
    low, high = low - margin, high + margin
    height = 7 * (high[1] - low[1]) / (high[0] - low[0])

    # Each fibre line as fractional sample positions on its own strip's grid.
    lines = [(case, line) for case, field in enumerate(across_fibres.to_array())
             for line in level_lines(field, spacing)]
    images = []
    for surface, grids in zip(surfaces, deformed):
        corners = np.stack([surface[:, :-1, :-1], surface[:, 1:, :-1], surface[:, 1:, 1:], surface[:, :-1, 1:]],
                           axis=-2).reshape(-1, 4, 3)                              # [quadrilaterals, 4, 3]
        normals = np.cross(corners[:, 2] - corners[:, 0], corners[:, 3] - corners[:, 1])
        normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
        shade = 0.5 + 0.5 * np.abs(normals @ LIGHT)
        flat, depth = projected(corners)
        order = np.argsort(-depth.mean(axis=-1))
        # The deformed surface, interpolated linearly between its samples at each fibre line's positions.
        threads = [np.stack([map_coordinates(grids[case][..., axis], line.T, order=1, mode="nearest") for axis in range(3)], axis=-1)
                   for case, line in lines]                                        # [lines] of [vertices, 3]
        figure = plt.figure(figsize=(7, height), dpi=DPI, facecolor="white")
        ax = figure.add_axes((0, 0, 1, 1))
        ax.add_collection(PolyCollection(flat[order], facecolors=FACE * shade[order, None],
                                         edgecolors=FACE * 0.8, linewidths=0.3))
        ax.add_collection(LineCollection([projected(thread)[0] for thread in threads],
                                         colors="#4a3a1f", linewidths=0.9))
        ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), aspect="equal")
        ax.set_axis_off()
        images.append(capture(figure))
        plt.close(figure)
    return images
