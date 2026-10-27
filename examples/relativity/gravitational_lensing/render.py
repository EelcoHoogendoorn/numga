"""Drawing for the binary lens: the source plane with the caustic, and the sky with the critical
curve as the zero level of the area ratio; and small round sources as the lens shows them on the sky."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from matplotlib.image import AxesImage

from examples.util.animation import capture
from examples.util.wgsl import pinned, room
from examples.relativity.gravitational_lensing import core

CRITICAL = "#55cbd3"
CAUSTIC = "#ff8772"
BRIGHTNESS = LinearSegmentedColormap.from_list(
    "starlight", ["#090d13", "#413043", "#966042", "#eab772", "#fff2ca"], N=256,
)
# The source unlensed, and the light the sky shows, each as a function of the direction.
Step = tuple[Callable[[core.Point], core.Scalar], Callable[[core.Point], core.Scalar]]
# Both panels show the directions within this distance of the optical axis.
HALF = 1.4
# Each panel's side and the white margin about it, in inches.
PANEL = 4.3
MARGIN = 0.1
# The still's resolution, and the animation's, below it: the blurred source needs no more.
STILL_DPI = 60
ANIMATION_DPI = 50


def xy(directions: core.Point) -> np.ndarray:
    """The x and y components of directions, their pairings with the planes x and y."""
    mv = directions.context.multivector
    return np.stack([(plane & directions).to_array() for plane in (mv.x, mv.y)], axis=-1)


def pixels(ax: plt.Axes, mv) -> core.Point:
    """The directions at the centres of the pixels the axes' room holds over the shown square, top row first."""
    resolution = min(room(ax))
    angles = (np.arange(resolution) + 0.5) * (2 * HALF / resolution) - HALF
    return (mv.x * angles[None, :] + mv.y * angles[::-1, None]).dual()           # [rows, columns] Point-


def critical_curve(ax: plt.Axes, directions: core.Point, area: core.Scalar, colour: str) -> list[core.Point]:
    """The critical curve, drawn as the zero level of the area ratio; returned as its pieces."""
    x, y = np.moveaxis(xy(directions), -1, 0)
    level = ax.contour(x, y, area.to_array(), levels=[0.0], colors=colour, linewidths=0.9)
    return [directions.context.multivector.vector(piece).dual() for piece in level.allsegs[0]]


def figure_at(directions: core.Point, area: core.Scalar, lens: Callable[[core.Point], core.Point],
              positions: core.Point, step: Step, dpi: int) -> tuple[plt.Figure, core.Point, list[AxesImage]]:
    """The source with the caustic, and its images on the sky with the critical curve and the masses, both over the
    same square, each picture pixel for pixel at the given resolution; with the pictures' pixels and the two images.
    The caustic is the critical curve carried to the source by the lens."""
    width, height = 2 * PANEL + 4 * MARGIN, PANEL + 2 * MARGIN
    figure, (source_ax, sky_ax) = plt.subplots(1, 2, figsize=(width, height), dpi=dpi, facecolor="white")
    figure.subplots_adjust(left=MARGIN / width, right=1 - MARGIN / width, bottom=MARGIN / height,
                           top=1 - MARGIN / height, wspace=2 * MARGIN / PANEL)
    shown = pixels(sky_ax, directions.context.multivector)                     # [rows, columns] Point-
    norm = PowerNorm(gamma=0.65, vmin=0, vmax=1)
    images = []
    for ax, light in zip((source_ax, sky_ax), step):
        image = pinned(ax, light(shown).to_array(), (-HALF, HALF, -HALF, HALF))
        image.set(cmap=BRIGHTNESS, norm=norm)
        images.append(image)
    for piece in critical_curve(sky_ax, directions, area, CRITICAL):
        source_ax.plot(*xy(lens(piece)).T, color=CAUSTIC, linewidth=0.9)
    sky_ax.scatter(*xy(positions).T, s=40, facecolors="none", edgecolors="0.65", linewidths=0.8)
    for ax in (source_ax, sky_ax):
        ax.set(xlim=(-HALF, HALF), ylim=(-HALF, HALF))
    return figure, shown, images


def draw(directions: core.Point, area: core.Scalar, lens: Callable[[core.Point], core.Point],
         positions: core.Point, step: Step) -> plt.Figure:
    """The still: the source and its images, the critical curve the zero level of the area ratio over the
    directions."""
    return figure_at(directions, area, lens, positions, step, STILL_DPI)[0]


def animate(directions: core.Point, area: core.Scalar, lens: Callable[[core.Point], core.Point],
            positions: core.Point, steps: Iterable[Step]) -> Iterator[np.ndarray]:
    """One frame per step: the figure is drawn once, and each step replaces its two images."""
    steps = iter(steps)
    figure, shown, images = figure_at(directions, area, lens, positions, next(steps), ANIMATION_DPI)
    yield capture(figure)
    for step in steps:
        for image, light in zip(images, step):
            image.set_data(light(shown).to_array())
        yield capture(figure)
    plt.close(figure)
