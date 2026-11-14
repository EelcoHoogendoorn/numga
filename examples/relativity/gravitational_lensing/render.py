"""Side-by-side rendering of the unlensed source with its caustic and the lensed sky with its critical curve."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, PowerNorm, to_rgb
from matplotlib.image import AxesImage

from numga import stack
from numga.render import simplices
from numga.topology import Topology
from examples.relativity.gravitational_lensing import core
from examples.util.animation import capture
from examples.util.panels import offset_panel, overlay, pinned, room

critical_colour = np.array(to_rgb("#55cbd3"))
caustic_colour = np.array(to_rgb("#ff8772"))
# Curve stroke width, lens-mass marker grey level, and marker radius in pixels.
curve_width = 1.0
mass_colour, mass_radius = np.full(3, 0.65), 2.5
brightness_cmap = LinearSegmentedColormap.from_list(
    "starlight", ["#090d13", "#413043", "#966042", "#eab772", "#fff2ca"], N=256,
)
# Half-width of both angular panels in Einstein radii, panel side length, margin in inches, and still/animation DPIs.
panel_half_width = 1.4
panel_inches = 4.3
margin_inches = 0.1
still_dpi = 60
animation_dpi = 50
# Row and column corner offsets around each pixel quad's four sides.
side_starts = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
side_ends = np.roll(side_starts, -1, axis=0)


def panel_directions(ax: plt.Axes, mv) -> core.Point:
    """Places sky directions (`[n_panel_pixels, n_panel_pixels] Point-`) at the centres of the display pixels inside `ax`, top row first."""
    n_panel_pixels = min(room(ax))
    angles = (np.arange(n_panel_pixels) + 0.5) * (2 * panel_half_width / n_panel_pixels) - panel_half_width
    return (mv.x * angles[None, :] + mv.y * angles[::-1, None]).dual()         # [n_panel_pixels, n_panel_pixels] Point-


def side_zero(
    directions: core.Point, area: core.Scalar, rows: np.ndarray, columns: np.ndarray, sides: np.ndarray,
) -> core.Point:
    """Interpolates the zero crossing (`[...] Point-`) of `area` linearly along each selected grid-quad side."""
    tails = rows + side_starts[sides, 0], columns + side_starts[sides, 1]
    heads = rows + side_ends[sides, 0], columns + side_ends[sides, 1]
    return (area[heads] * directions[tails] - area[tails] * directions[heads]) / (area[heads] - area[tails])   # [...] Point-


def critical_curve(directions: core.Point, area: core.Scalar) -> tuple[core.Point, core.Point]:
    """Extracts the segment endpoints (`[...] Point-`, `[...] Point-`) where `area` changes sign across grid quads."""
    quad_indices = np.indices(np.array(area.shape) - 1)[:, None]               # [2, 1, ...]
    tails = tuple(quad_indices + side_starts.T[..., None, None])
    heads = tuple(quad_indices + side_ends.T[..., None, None])
    positive = area > 0
    crossed = positive[tails] != positive[heads]                               # [4, ...]
    # Order crossed sides first around each quad; two crossings form one segment and four crossings at a saddle form two.
    order = np.argsort(~crossed, axis=0, kind="stable")                        # [4, ...]
    pieces, rows, columns = np.nonzero(crossed.sum(axis=0) > 2 * np.arange(2)[:, None, None])
    return (
        side_zero(directions, area, rows, columns, order[2 * pieces, rows, columns]),
        side_zero(directions, area, rows, columns, order[2 * pieces + 1, rows, columns]),
    )


def lensing_layout(
    directions: core.Point,
    area: core.Scalar,
    lens: Callable[[core.Point], core.Point],
    positions: core.Point,
    exposure: core.Exposure,
    dpi: int,
) -> tuple[plt.Figure, core.Point, list[AxesImage]]:
    """Builds the two-panel source and sky figure at `dpi`, returning the figure, display pixel grid, and image artists."""
    width, height = 2 * panel_inches + 4 * margin_inches, panel_inches + 2 * margin_inches
    figure, (source_ax, sky_ax) = plt.subplots(1, 2, figsize=(width, height), dpi=dpi, facecolor="white")
    figure.subplots_adjust(
        left=margin_inches / width,
        right=1 - margin_inches / width,
        bottom=margin_inches / height,
        top=1 - margin_inches / height,
        wspace=2 * margin_inches / panel_inches,
    )
    shown_directions = panel_directions(sky_ax, directions.context.multivector)   # [..., ...] Point-
    norm = PowerNorm(gamma=0.65, vmin=0, vmax=1)
    bounds = (-panel_half_width, panel_half_width, -panel_half_width, panel_half_width)
    images = []
    for ax, light in zip((source_ax, sky_ax), exposure):
        image = pinned(ax, light(shown_directions).to_array(), bounds)
        image.set(cmap=brightness_cmap, norm=norm)
        images.append(image)
    for ax in (source_ax, sky_ax):
        ax.set(xlim=(-panel_half_width, panel_half_width), ylim=(-panel_half_width, panel_half_width))
    # Map the critical curve's sky segments forward through the lens to draw the caustic on the source panel.
    segment_ends = stack(critical_curve(directions, area), axis=-1).reshape(-1)   # [...] Point-
    n_segments = segment_ends.shape[0] // 2
    chains = Topology.chains([2] * n_segments)
    sky_camera = offset_panel(sky_ax, core.mv.x, core.mv.y)
    source_camera = offset_panel(source_ax, core.mv.x, core.mv.y)
    overlay(
        sky_ax,
        [
            simplices.strokes(segment_ends, chains, critical_colour, curve_width, sky_camera),
            simplices.markers(positions, mass_colour, mass_radius, sky_camera),
        ],
    )
    overlay(source_ax, [simplices.strokes(lens(segment_ends), chains, caustic_colour, curve_width, source_camera)])
    return figure, shown_directions, images


def lensing_figure(
    directions: core.Point,
    area: core.Scalar,
    lens: Callable[[core.Point], core.Point],
    positions: core.Point,
    exposure: core.Exposure,
) -> plt.Figure:
    """Draws the static two-panel figure showing the unlensed source with its caustic and the lensed sky with its critical curve."""
    return lensing_layout(directions, area, lens, positions, exposure, still_dpi)[0]


def lensing_animation(
    directions: core.Point,
    area: core.Scalar,
    lens: Callable[[core.Point], core.Point],
    positions: core.Point,
    exposures: Iterable[core.Exposure],
) -> Iterator[np.ndarray]:
    """Yields RGB frames (`Iterator[np.ndarray]`) updating both brightness panels as the source moves across the caustic."""
    exposure_iter = iter(exposures)
    figure, shown_directions, images = lensing_layout(
        directions, area, lens, positions, next(exposure_iter), animation_dpi,
    )
    yield capture(figure)
    for exposure in exposure_iter:
        for image, light in zip(images, exposure):
            image.set_data(light(shown_directions).to_array())
        yield capture(figure)
    plt.close(figure)
