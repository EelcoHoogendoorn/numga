"""Matplotlib axes layout and camera helpers place GPU-rendered `[rows, columns, ...]` images into
figures pixel for pixel without resampling.

`pinned`, `show`, `show_pixels`, and `overlay` attach an axes locator that centres an image on
integer canvas pixels at the figure's native `dpi` (and scales each source pixel into an exact
integer block under `animation.capture` supersampling), while keeping matplotlib data coordinates
aligned with the image's world or pixel `extent` for text annotations and overlays.

`panel` and `offset_panel` size a `numga.render.Camera` to the pixel budget `room(ax)` of a
matplotlib `Axes` and frame it to the axes' `xlim` and `ylim`. Projective scenes supply an explicit
`weight` plane (`panel`), whereas vector-space scenes treat points as unit-weight offsets from the
origin (`offset_panel`). Companion notes on reading points through regressive pairings with planes
are in `docs/orientation.md`.
"""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes

from numga.extensor import Extensor
from numga.render.camera import Camera
from numga.render.drawing import composited


def pinned(ax, image: np.ndarray, extent: tuple[float, float, float, float]):
    """Displays `image` (`[rows, columns, ...]`) over `(left, right, bottom, top)` `extent` with
    axes hidden, returning the `AxesImage` artist so animations can update its pixel data.

    An axes locator centres the image inside the layout box allocated to `ax` on whole canvas pixels
    at the figure's base `dpi`, avoiding resampling at base resolution and mapping each source pixel
    to an integer block when `capture` supersamples the figure.
    """
    resolution = ax.figure.figure.dpi
    rows, columns = image.shape[:2]

    def exact(ax, renderer):
        from matplotlib.transforms import Bbox

        figure = ax.figure.figure
        scale = figure.dpi / resolution
        # Read the allocated box in root-canvas pixels through `ax.figure`, which may be a SubFigure.
        allocated = ax.get_position(original=True).transformed(ax.figure.transSubfigure)
        # Snap the top-left corner to a whole pixel at the base resolution so integer supersampling
        # lands on whole pixels as well.
        canvas_rows = int(figure.bbox.height)
        left = round(((allocated.x0 + allocated.x1) / 2 - columns * scale / 2) / scale) * scale
        top = round((canvas_rows - (allocated.y0 + allocated.y1) / 2 - rows * scale / 2) / scale) * scale
        bottom = canvas_rows - top - rows * scale
        # Nudge the origin slightly up and the span slightly down so matplotlib's floor/ceil pixel
        # rounding recovers the exact integer bounds after the normalized subfigure round trip.
        pixels = Bbox.from_bounds(left + 1e-6, bottom + 1e-6, columns * scale - 1e-6, rows * scale - 1e-6)
        return pixels.transformed(ax.figure.transSubfigure.inverted())

    shown = ax.imshow(image, extent=extent, interpolation="nearest", aspect="auto")
    ax.set_axes_locator(exact)
    ax.set_axis_off()
    return shown


def room(ax) -> tuple[int, int]:
    """Executing the figure's layout engine determines the `(columns, rows)` pixel budget allocated
    to `ax` at the root figure's `dpi`."""
    # Run layout on the root figure, then read `ax` in canvas pixels through its immediate figure or subfigure.
    figure = ax.figure.figure
    if figure.get_layout_engine() is not None:
        figure.get_layout_engine().execute(figure)
    box = ax.get_position(original=True).transformed(ax.figure.transSubfigure)
    return int(box.width), int(box.height)


def fitted(view, ax):
    """Replaces `view.pixels` with the largest horizontal pixel count whose aspect-matched image
    fits completely inside `room(ax)`."""
    columns, rows = room(ax)
    return view._replace(pixels=min(columns, int(rows * view.half[0] / view.half[1])))


def show(ax, image: np.ndarray, view):
    """Pins `image` (`[rows, columns, ...]`) into `ax` over `view.extent` in scene coordinates so
    projected points from `view.across` align with the displayed pixels, returning the `AxesImage`."""
    return pinned(ax, image, view.extent)


def show_pixels(ax, image: np.ndarray):
    """Pins `image` (`[rows, columns, ...]`) into `ax` in pixel coordinates with row `0` at the top,
    returning the `AxesImage`."""
    rows, columns = image.shape[:2]
    return pinned(ax, image, (-0.5, columns - 0.5, rows - 0.5, -0.5))


def labels(ax, view, places, texts: list[str], **style) -> None:
    """Projects `places` (`[n_places]` points of any weight) through `view.across` and draws `texts`
    at their projected coordinates on `ax`."""
    for (x, y), text in zip(view.across(places), texts):
        ax.text(x, y, text, **style)


def image_figure(image: np.ndarray):
    """Wraps `image` (`[rows, columns, ...]`) in a borderless `100`-dpi matplotlib `Figure` matching
    its exact pixel dimensions."""
    import matplotlib.pyplot as plt

    figure, ax = plt.subplots(figsize=(image.shape[1] / 100, image.shape[0] / 100), dpi=100)
    ax.imshow(image, interpolation="nearest")
    ax.set_axis_off()
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
    return figure


def limits(ax: Axes) -> tuple[np.ndarray, np.ndarray]:
    """Reads `ax`'s `xlim` and `ylim` into `(centre, half_span)` arrays, each of shape `[2]`."""
    left, right = ax.get_xlim()
    bottom, top = ax.get_ylim()
    return np.array([left + right, bottom + top]) / 2, np.array([right - left, top - bottom]) / 2


def panel(ax: Axes, across: Extensor, up: Extensor, weight: Extensor) -> Camera:
    """Frames `ax`'s data limits with a projective `Camera` that reads horizontal and vertical
    coordinates with `across` and `up` divided by `weight & point`, fitted to `room(ax)`."""
    return fitted(Camera(across, up, 0 * across, weight, 0.0, *limits(ax), 300), ax)


def offset_panel(ax: Axes, across: Extensor, up: Extensor) -> Camera:
    """Frames `ax`'s data limits with an affine `Camera` that reads unit-weight origin offsets with
    `across` and `up`, fitted to `room(ax)`."""
    return fitted(Camera(across, up, 0 * across, 0 * across, 1.0, *limits(ax), 300), ax)


def overlay(ax: Axes, pictures: list, *, zorder: float | None = None):
    """Composites RGBA `pictures` (`[rows, columns, 4]`) in order and pins the result over `ax`'s
    current `xlim` and `ylim` with transparent uncovered pixels, returning the `AxesImage`."""
    shown = pinned(ax, composited(pictures), (*ax.get_xlim(), *ax.get_ylim()))
    if zorder is not None:
        shown.set_zorder(zorder)
    return shown

