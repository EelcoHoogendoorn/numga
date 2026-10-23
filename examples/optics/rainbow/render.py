"""Drawing for the rainbow: light swept across a raindrop beside the angles it leaves at, and the double
bow on the sky opposite the sun, seen from the ground."""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from numga import stack
from examples.util.animation import capture
from examples.util.wgsl import pinned, room
from examples.optics.rainbow import core

mv = core.mv
# The colours of the drop, of light reflected once and twice inside it, of the sky and of the ground.
BACKGROUND = "#0b1020"
DROP = "#1e3a5f"
SUNLIGHT = "#fde68a"
PRIMARY = "#fb923c"
SECONDARY = "#60a5fa"
SKY = np.array([0.04, 0.06, 0.12])
GROUND = "#14532d"
# How far light is drawn outside the drop, in drop radii; the width of the angles the sky's light is
# gathered in, and the half width of the view of the sky, in degrees.
REACH = 2.4
BIN = 0.05
VIEW = 62.0
# The power the sky's light is raised to for display, brightening faint light more than bright.
TONE = 0.65


# --- plumbing -------------------------------------------------------------------------
def flat(points: core.Direction) -> np.ndarray:
    """Points of the cross-section through the drop along the sunlight, on the last axis: along the
    sunlight, and up."""
    return (stack((mv.x, mv.y)) & points[..., None]).to_array()


def pixel_grid(view: tuple[float, float, float, float], columns: int, rows: int
               ) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    """The middles of the square pixels of the view, left, right, bottom and top, that fit in `columns` by `rows`:
    across, and up with the rows down from the top; and the grid's extent, the outer edges of its pixels."""
    left, right, bottom, top = view
    pixel = max((right - left) / columns, (top - bottom) / rows)
    n_columns, n_rows = round((right - left) / pixel), round((top - bottom) / pixel)
    middle, half = np.array([left + right, bottom + top]) / 2, np.array([n_columns, n_rows]) * pixel / 2
    across = middle[0] - half[0] + pixel * (np.arange(n_columns) + 0.5)
    up = middle[1] + half[1] - pixel * (np.arange(n_rows) + 0.5)
    return across, up, (middle[0] - half[0], middle[0] + half[0], middle[1] - half[1], middle[1] + half[1])


def degrees(apart: core.Scalar) -> np.ndarray:
    """The angle from the point opposite the sun, in degrees, from its cosine."""
    return np.degrees(np.arccos(np.clip(apart.to_array(), -1, 1)))


def spectrum(wavelengths: np.ndarray) -> np.ndarray:
    """The colour of light of each wavelength `[colours, 3]`, an approximation of the spectral colours,
    scaled so that light of every wavelength together is white."""
    red = np.interp(wavelengths, [380, 440, 490, 510, 580, 645, 780], [0.4, 0, 0, 0, 1, 1, 0.5])
    green = np.interp(wavelengths, [380, 440, 490, 510, 580, 645, 780], [0, 0, 1, 1, 1, 0, 0])
    blue = np.interp(wavelengths, [380, 440, 490, 510, 580, 645, 780], [0.6, 1, 1, 0, 0, 0, 0])
    colours = np.stack([red, green, blue], axis=-1)
    return colours / colours.sum(axis=0)


def brightness(apart: core.Scalar, flux: core.Scalar, sun_width: float) -> tuple[np.ndarray, np.ndarray]:
    """The angles from the point opposite the sun, and the light per solid angle seen at each, per
    colour `[colours, angles]`: the light gathered into narrow rings, spread over the sun's disc."""
    edges = np.arange(0, 90 + BIN, BIN)
    angles = (edges[1:] + edges[:-1]) / 2
    seen, weights = degrees(apart), flux.to_array()                          # [orders, colours, rays]
    gathered = np.stack([np.histogram(seen[:, colour].ravel(), edges, weights=weights[:, colour].ravel())[0]
                         for colour in range(seen.shape[1])])                 # [colours, angles]
    # Light from each point of the sun's disc arrives from a slightly different direction.
    across = np.arange(-sun_width / 2, sun_width / 2 + BIN, BIN)
    disc = np.sqrt(np.clip(1 - (2 * across / sun_width) ** 2, 0, None))
    spread = np.stack([np.convolve(ring, disc / disc.sum(), mode="same") for ring in gathered])
    return angles, spread / np.sin(np.radians(angles))


def draw_sky(apart: core.Scalar, flux: core.Scalar, wavelengths: np.ndarray, sun_elevation: float,
             sun_width: float) -> plt.Figure:
    """The sky opposite the sun, with the point opposite the sun below the horizon by the sun's height."""
    angles, light = brightness(apart, flux, sun_width)
    colour = light.T @ spectrum(wavelengths)                                  # [angles, 3]
    colour = colour / colour.max()
    # The view of the sky centred above the point opposite the sun, from below the horizon, in degrees.
    view = (-VIEW, VIEW, sun_elevation - 6, sun_elevation + VIEW * 0.8)
    figure = plt.figure(figsize=(9.0, 9.0 * (view[3] - view[2]) / (view[1] - view[0])), dpi=100, facecolor=BACKGROUND)
    ax = figure.add_axes((0, 0, 1, 1))
    # Each pixel the figure shows at its angle from the point opposite the sun.
    across, height, extent = pixel_grid(view, *room(ax))
    distance = np.hypot(*np.meshgrid(across, height))
    image = np.stack([np.interp(distance, angles, colour[:, channel], right=0) for channel in range(3)], axis=-1)
    pinned(ax, np.clip(SKY + image ** TONE, 0, 1), extent)
    ax.fill_between(extent[:2], extent[2], sun_elevation, color=GROUND)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    return figure


def animate_sweep(swept: tuple[core.Direction, core.Direction, core.Direction, core.Scalar],
                  bundle: tuple[core.Direction, core.Direction, core.Direction, core.Scalar],
                  offsets: core.Direction, apart: core.Scalar, wavelengths: np.ndarray) -> Iterator[np.ndarray]:
    """Light swept across the drop, with a bundle of light behind it, its paths after one and two
    reflections; beside it, the angle each colour is seen at against the offset of the light."""
    def rays(entry, vertices, exits):
        """Per order drawn, the path of each ray from the sun to past the drop `[orders, rays, points, 2]`."""
        start, inside, ahead = flat(entry), flat(vertices), flat(vertices + REACH * exits)
        sunward = start - np.array([REACH, 0])
        return [np.concatenate([np.stack([sunward, start], axis=-2), np.moveaxis(inside[:order + 1], 0, -2),
                                ahead[order][..., None, :]], axis=-2) for order in (1, 2)]

    swept_rays, bundle_rays = rays(*swept[:3]), rays(*bundle[:3])
    heights = (mv.y & offsets).to_array()                                     # [rays]
    curves, colours = degrees(apart), spectrum(wavelengths) / spectrum(wavelengths).max(axis=-1, keepdims=True)
    marks, swept_heights = degrees(swept[3]), (mv.y & swept[0]).to_array()   # [orders + 1, frames], [frames]
    # The drop, the bundle of light behind it and each colour's angles stay put; the swept light and its marks move.
    figure = plt.figure(figsize=(11.0, 5.0), dpi=100, facecolor=BACKGROUND)
    drop = figure.add_axes((0, 0, 0.5, 1))
    drop.set_facecolor(BACKGROUND)
    drop.add_patch(plt.Circle((0, 0), 1, color=DROP))
    for paths, colour in zip(bundle_rays, (PRIMARY, SECONDARY)):
        drop.add_collection(LineCollection(paths[:, -2:], colors=colour, linewidths=0.6, alpha=0.4))
    # The swept light's paths after two reflections and after one, and the sunlight reaching the drop.
    lights = [drop.plot([], [], color=colour, linewidth=2)[0] for colour in (SECONDARY, PRIMARY, SUNLIGHT)]
    drop.set_xlim(-REACH, REACH)
    drop.set_ylim(-REACH * 0.9, REACH * 0.9)
    drop.set_aspect("equal")
    drop.set_axis_off()
    angles = figure.add_axes((0.57, 0.12, 0.4, 0.82))
    angles.set_facecolor(BACKGROUND)
    markers = []
    for order, mark in ((0, PRIMARY), (1, SECONDARY)):
        for curve, colour in zip(curves[order], colours):
            angles.plot(heights, curve, color=colour, linewidth=0.8)
        markers += angles.plot([], [], "o", color=mark, markersize=8)
    angles.set_xlim(0, 1)
    angles.set_ylim(0, 90)
    angles.tick_params(colors="#94a3b8")
    for spine in angles.spines.values():
        spine.set_color("#475569")
    for frame in range(len(swept_heights)):
        for light, path in zip(lights, (swept_rays[1][frame, 1:], swept_rays[0][frame, 1:], swept_rays[0][frame, :2])):
            light.set_data(*path.T)
        for order, marker in enumerate(markers):
            marker.set_data([swept_heights[frame]], [marks[order + 1, frame]])
        yield capture(figure)
    plt.close(figure)
