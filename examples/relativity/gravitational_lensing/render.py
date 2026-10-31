"""Drawing for the binary lens: the source plane with the caustic, and the sky with the critical
curve as the zero level of the area ratio; and small round sources as the lens shows them on the sky."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, PowerNorm, to_rgb
from matplotlib.image import AxesImage

from numga import stack
from examples.util.animation import capture
from examples.util.wgsl import pinned, room, simplices
from examples.util.wgsl.camera import flat_panel
from examples.relativity.gravitational_lensing import core

CRITICAL = np.array(to_rgb("#55cbd3"))
CAUSTIC = np.array(to_rgb("#ff8772"))
# The curves' width, and the masses' rings: their ink, and how far they reach from their points, in pixels.
CURVE_WIDTH = 1.0
MASS, MASS_OUTER, MASS_INNER = np.full(3, 0.65), 3.0, 2.0
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
# The corners of a grid cell, as offsets in rows and columns, that each of its sides runs from and to, around it.
side_starts = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
side_ends = np.roll(side_starts, -1, axis=0)


def pixels(ax: plt.Axes, mv) -> core.Point:
    """The directions at the centres of the pixels the axes' room holds over the shown square, top row first."""
    resolution = min(room(ax))
    angles = (np.arange(resolution) + 0.5) * (2 * HALF / resolution) - HALF
    return (mv.x * angles[None, :] + mv.y * angles[::-1, None]).dual()           # [rows, columns] Point-


def zero_along(directions: core.Point, area: core.Scalar, rows: np.ndarray, columns: np.ndarray,
               sides: np.ndarray) -> core.Point:
    """The direction on each given side of the given cells of the grid where the area ratio, linear along the side,
    vanishes: the side's two ends weighted by the ratio at the other end."""
    tails = rows + side_starts[sides, 0], columns + side_starts[sides, 1]
    heads = rows + side_ends[sides, 0], columns + side_ends[sides, 1]
    return (area[heads] * directions[tails] - area[tails] * directions[heads]) / (area[heads] - area[tails])


def critical_curve(directions: core.Point, area: core.Scalar) -> tuple[core.Point, core.Point]:
    """The critical curve as straight pieces across the cells of the grid of directions, their two ends: the zeros of
    the area ratio on the sides of each cell it crosses, joined in pairs."""
    # The row and column indices of the corners each side of each cell runs from and to, [sides, rows - 1, columns - 1].
    cells = np.indices(np.array(area.shape) - 1)[:, None]                      # [2, 1, rows - 1, columns - 1]
    tails = tuple(cells + side_starts.T[..., None, None])
    heads = tuple(cells + side_ends.T[..., None, None])
    positive = area > 0
    crossed = positive[tails] != positive[heads]                               # [sides, rows - 1, columns - 1]
    # The crossed sides of each cell first, in their order around it. A cell holds one piece where the curve crosses
    # two of its sides, and two at a saddle, where it crosses all four.
    order = np.argsort(~crossed, axis=0, kind="stable")                        # [sides, rows - 1, columns - 1]
    pieces, rows, columns = np.nonzero(crossed.sum(axis=0) > 2 * np.arange(2)[:, None, None])
    return (zero_along(directions, area, rows, columns, order[2 * pieces, rows, columns]),
            zero_along(directions, area, rows, columns, order[2 * pieces + 1, rows, columns]))


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
    for ax in (source_ax, sky_ax):
        ax.set(xlim=(-HALF, HALF), ylim=(-HALF, HALF))
    # The critical curve's straight pieces, each from its tail to its head, over the sky, with the masses' rings; the
    # caustic, the same pieces carried by the lens, over the source.
    ends = stack(critical_curve(directions, area), axis=-1).reshape(-1)        # [pieces * 2] Point-
    pieces = simplices.Mesh.chains([2] * (ends.shape[0] // 2))
    sky = flat_panel(sky_ax, ends.algebra)
    simplices.overlay(sky_ax, [simplices.strokes(ends, pieces, CRITICAL, CURVE_WIDTH, sky),
                               simplices.markers(positions, MASS, MASS_OUTER, sky, inner=MASS_INNER)])
    simplices.overlay(source_ax, [simplices.strokes(lens(ends), pieces, CAUSTIC, CURVE_WIDTH,
                                                    flat_panel(source_ax, ends.algebra))])
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
