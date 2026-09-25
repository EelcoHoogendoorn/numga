"""Drawing for the binary lens: the source plane with the caustic, and the sky with the critical
curve as the zero level of the area ratio; and small round sources as the lens shows them on the sky."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, PowerNorm

from examples.animation import capture
from examples.relativity.gravitational_lensing import core

CRITICAL = "#55cbd3"
CAUSTIC = "#ff8772"
PRESERVED = "#267fa9"
REVERSED = "#c96948"
BRIGHTNESS = LinearSegmentedColormap.from_list(
    "starlight", ["#090d13", "#413043", "#966042", "#eab772", "#fff2ca"], N=256,
)
# The source unlensed, and the light the sky shows.
Step = tuple[core.Scalar, core.Scalar]
# Both panels show the directions within this distance of the optical axis.
HALF = 1.4


def xy(vectors: core.Vector) -> np.ndarray:
    """The x and y components of directions, read in their own algebra."""
    mv = vectors.context.multivector
    return np.stack([(vectors | axis).to_array() for axis in (mv.x, mv.y)], axis=-1)


def sky_axes(ax: plt.Axes, title: str) -> None:
    """Equal axes over the square both panels share, without ticks."""
    ax.set(xlim=(-HALF, HALF), ylim=(-HALF, HALF), aspect="equal", xticks=[], yticks=[], title=title)


def critical_curve(ax: plt.Axes, directions: core.Vector, area: core.Scalar, colour: str) -> list[core.Vector]:
    """The critical curve, drawn as the zero level of the area ratio; returned as its pieces."""
    x, y = np.moveaxis(xy(directions), -1, 0)
    level = ax.contour(x, y, area.to_array(), levels=[0.0], colors=colour, linewidths=0.9)
    return [directions.context.multivector.vector(piece) for piece in level.allsegs[0]]


def draw(directions: core.Vector, area: core.Scalar, lens: Callable[[core.Vector], core.Vector],
         positions: core.Vector, step: Step) -> plt.Figure:
    """The source with the caustic, and its images on the sky with the critical
    curve and the masses, both over the same directions and at the same scale. The caustic is the
    critical curve carried to the source by the lens."""
    figure, (source_ax, sky_ax) = plt.subplots(1, 2, figsize=(9, 4.6), facecolor="white")
    source_light, image_light = step
    norm = PowerNorm(gamma=0.65, vmin=0, vmax=1)
    corners = xy(directions[[0, -1], [0, -1]])                                 # [2, 2]
    box = [corners[0, 0], corners[1, 0], corners[0, 1], corners[1, 1]]
    for ax, light in ((source_ax, source_light), (sky_ax, image_light)):
        ax.imshow(light.to_array(), extent=box, origin="lower", cmap=BRIGHTNESS, norm=norm, interpolation="bilinear")
    for piece in critical_curve(sky_ax, directions, area, CRITICAL):
        source_ax.plot(*xy(lens(piece)).T, color=CAUSTIC, linewidth=0.9)
    sky_ax.scatter(*xy(positions).T, s=40, facecolors="none", edgecolors="0.65", linewidths=0.8)
    sky_axes(source_ax, "source")
    sky_axes(sky_ax, "sky")
    figure.tight_layout()
    return figure


def animate(directions: core.Vector, area: core.Scalar, lens: Callable[[core.Vector], core.Vector],
            positions: core.Vector, steps: Iterable[Step]) -> list[np.ndarray]:
    """One frame per step."""
    frames = []
    for step in steps:
        figure = draw(directions, area, lens, positions, step)
        frames.append(capture(figure))
        plt.close(figure)
    return frames


def draw_tissot(directions: core.Vector, area: core.Scalar, positions: core.Vector, ellipses: core.Vector,
                ratios: core.Scalar) -> plt.Figure:
    """Small round sources as they appear on the sky, about their sightlines: blue where the image
    keeps its orientation, red where it is reversed; with the critical curve and the masses."""
    figure, ax = plt.subplots(figsize=(6, 6), facecolor="white")
    outlines = xy(ellipses).reshape(-1, ellipses.shape[-1], 2)                # [ellipses, angles, 2]
    kept = ratios.to_array().reshape(-1) >= 0                                  # [ellipses]
    for outline, keeps in zip(outlines, kept):
        colour = PRESERVED if keeps else REVERSED
        ax.fill(*outline.T, color=colour, alpha=0.35, linewidth=0)
        ax.plot(*outline.T, color=colour, linewidth=0.8)
    critical_curve(ax, directions, area, "0.3")
    ax.scatter(*xy(positions).T, s=40, facecolors="none", edgecolors="0.4", linewidths=0.8)
    sky_axes(ax, "small round sources, as seen")
    return figure
