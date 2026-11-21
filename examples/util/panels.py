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

from numga import dim, stack
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


def _panel_camera(
    ax: Axes,
    across: Extensor,
    up: Extensor,
    weight: Extensor,
    unit: float,
    xlim: Extensor | None,
    ylim: Extensor | None,
) -> Camera:
    """Build a `Camera` fitted to `room(ax)`, scaling checked `across` and `up` planes into `dim.pixel`."""
    if xlim is not None or ylim is not None:
        if xlim is None or ylim is None:
            raise ValueError("xlim and ylim must both be supplied together")
        x_arr = np.asarray(xlim.to_array(), dtype=float).reshape(2)
        y_arr = np.asarray(ylim.to_array(), dtype=float).reshape(2)
        ax.set_xlim(float(x_arr[0]), float(x_arr[1]))
        ax.set_ylim(float(y_arr[0]), float(y_arr[1]))

    if not across.has_units:
        return fitted(Camera(across, up, 0 * across, weight, unit, *limits(ax), 300), ax)

    mv = across.context.multivector
    columns, rows = room(ax)
    if xlim is not None and ylim is not None:
        span_x, span_y = xlim[1] - xlim[0], ylim[1] - ylim[0]
        mid_x, mid_y = (xlim[0] + xlim[1]) / 2, (ylim[0] + ylim[1]) / 2
    else:
        mid, half = limits(ax)
        span_x, span_y = mv.scalar(float(2 * half[0])), mv.scalar(float(2 * half[1]))
        mid_x, mid_y = mv.scalar(float(mid[0])), mv.scalar(float(mid[1]))

    same_unit = (
        span_x.units.is_zero
        or span_y.units.is_zero
        or span_x.unit == span_y.unit
    )
    if same_unit:
        aspect = float((span_x / span_y).to_array())
        fitted_columns = min(columns, int(rows * aspect))
        scale_x = mv.scalar(float(fitted_columns), unit=dim.pixel) / span_x
        scale_y = scale_x
    else:
        scale_x = mv.scalar(float(columns), unit=dim.pixel) / span_x
        scale_y = mv.scalar(float(rows), unit=dim.pixel) / span_y

    across_px, up_px = scale_x * across, scale_y * up
    middle_px = stack((scale_x * mid_x, scale_y * mid_y))
    half_px = stack((scale_x * (span_x / 2), scale_y * (span_y / 2)))
    return Camera(across_px, up_px, 0 * across_px, weight, unit, middle_px, half_px)


def panel(ax: Axes, across: Extensor, up: Extensor, weight: Extensor, *,
          xlim: Extensor | None = None, ylim: Extensor | None = None) -> Camera:
    """Frames `ax`'s data limits with a projective `Camera` that reads horizontal and vertical
    coordinates with `across` and `up` divided by `weight & point`, fitted to `room(ax)`. When
    `across` carries checked units, optional `xlim`/`ylim` (`[2] Scalar+`) set the axes limits and
    scale the camera planes into `dim.pixel`."""
    return _panel_camera(ax, across, up, weight, 0.0, xlim, ylim)


def offset_panel(ax: Axes, across: Extensor, up: Extensor, *,
                 xlim: Extensor | None = None, ylim: Extensor | None = None) -> Camera:
    """Frames `ax`'s data limits with an affine `Camera` that reads unit-weight origin offsets with
    `across` and `up`, fitted to `room(ax)`. When `across` carries checked units, optional
    `xlim`/`ylim` (`[2] Scalar+`) set the axes limits and scale the camera planes into `dim.pixel`."""
    weight = across.context.multivector.zeros(across.gatype.derive.unchecked) if across.has_units else 0 * across
    return _panel_camera(ax, across, up, weight, 1.0, xlim, ylim)


def overlay(ax: Axes, pictures: list, *, zorder: float | None = None):
    """Composites RGBA `pictures` (`[rows, columns, 4]`) in order and pins the result over `ax`'s
    current `xlim` and `ylim` with transparent uncovered pixels, returning the `AxesImage`."""
    shown = pinned(ax, composited(pictures), (*ax.get_xlim(), *ax.get_ylim()))
    if zorder is not None:
        shown.set_zorder(zorder)
    return shown

