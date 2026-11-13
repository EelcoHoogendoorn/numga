"""Matplotlib figure and panel helpers for pinning GPU-rendered images pixel for pixel, fitting cameras and views to
axes, and overlaying composited layers."""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes

from numga.extensor import Extensor
from numga.render.camera import Camera
from numga.render.simplices import composited


def pinned(ax, image: np.ndarray, extent: tuple[float, float, float, float]):
    """An image in matplotlib axes over an extent, left, right, bottom and top, without axes; the image, for an
    animation to set its data.

    The axes hold the image pixel for pixel: whatever room the figure's layout gives them, they are drawn exactly as
    many pixels across and up as the image, at the figure's own resolution, about the middle of that room and on whole
    pixels, so that no pixel of the picture is resampled. Drawn finer, as `capture` draws, each pixel is a whole block
    of them."""
    resolution = ax.figure.figure.dpi
    rows, columns = image.shape[:2]

    def exact(ax, renderer):
        """The axes' place in the figure: the image's own pixels, about the middle of the room the layout gave them."""
        from matplotlib.transforms import Bbox

        figure = ax.figure.figure
        scale = figure.dpi / resolution
        # The room in pixels of the canvas, read through the axes' own figure, a subfigure too.
        room = ax.get_position(original=True).transformed(ax.figure.transSubfigure)
        # The top left corner on a whole pixel of the figure's own resolution, and so of any multiple of it, counted
        # from the top left of the canvas, as its rows of pixels are.
        canvas_rows = int(figure.bbox.height)
        left = round(((room.x0 + room.x1) / 2 - columns * scale / 2) / scale) * scale
        top = round((canvas_rows - (room.y0 + room.y1) / 2 - rows * scale / 2) / scale) * scale
        bottom = canvas_rows - top - rows * scale
        # The corner a millionth of a pixel over its whole pixel and the size a millionth under, so that the round
        # trip through the figure's fractions lands on them as matplotlib rounds them: the corner down, the size up.
        pixels = Bbox.from_bounds(left + 1e-6, bottom + 1e-6, columns * scale - 1e-6, rows * scale - 1e-6)
        return pixels.transformed(ax.figure.transSubfigure.inverted())

    shown = ax.imshow(image, extent=extent, interpolation="nearest", aspect="auto")
    ax.set_axes_locator(exact)
    ax.set_axis_off()
    return shown


def room(ax) -> tuple[int, int]:
    """The columns and rows of pixels the figure's layout gives the axes, at the figure's own resolution: the size a
    picture shown in them is drawn at to fill them."""
    # The root figure holds the layout; the axes' place is read in pixels through their own figure, a subfigure too.
    figure = ax.figure.figure
    if figure.get_layout_engine() is not None:
        figure.get_layout_engine().execute(figure)
    box = ax.get_position(original=True).transformed(ax.figure.transSubfigure)
    return int(box.width), int(box.height)


def fitted(view, ax):
    """The view with as many pixels across as the room its axes get holds, the whole picture inside the room, for a
    picture `show` holds pixel for pixel."""
    columns, rows = room(ax)
    return view._replace(pixels=min(columns, int(rows * view.half[0] / view.half[1])))


def show(ax, image: np.ndarray, view):
    """An image of a view in matplotlib axes, its middle at the origin, in the scene's units, so that places from
    `view.across` land where they are seen, held pixel for pixel as `pinned` holds it; the image, for an animation to
    set its data."""
    return pinned(ax, image, view.extent)


def show_pixels(ax, image: np.ndarray):
    """An image in matplotlib axes in its own pixels, rows down from the top, held pixel for pixel as `pinned` holds
    it; the image, for an animation to set its data."""
    rows, columns = image.shape[:2]
    return pinned(ax, image, (-0.5, columns - 0.5, rows - 0.5, -0.5))


def labels(ax, view, places, texts: list[str], **style) -> None:
    """Texts at points, of any weight, as a view sees them, over its image shown by `show`."""
    for (x, y), text in zip(view.across(places), texts):
        ax.text(x, y, text, **style)


def image_figure(image: np.ndarray):
    """A figure of an image at its pixel size, without axes or margins."""
    import matplotlib.pyplot as plt

    figure, ax = plt.subplots(figsize=(image.shape[1] / 100, image.shape[0] / 100), dpi=100)
    ax.imshow(image, interpolation="nearest")
    ax.set_axis_off()
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
    return figure


def limits(ax: Axes) -> tuple[np.ndarray, np.ndarray]:
    """The readings across and up at the middle of a matplotlib panel's limits, and half its width and height in
    them."""
    left, right = ax.get_xlim()
    bottom, top = ax.get_ylim()
    return np.array([left + right, bottom + top]) / 2, np.array([right - left, top - bottom]) / 2


def panel(ax: Axes, across: Extensor, up: Extensor, weight: Extensor) -> Camera:
    """The camera of a matplotlib panel reading its points by the planes `across` and `up` over their weight, their
    pairing with the plane `weight`, such as a projective algebra's plane at infinity; its limits the readings it spans,
    as many pixels across as the panel's room holds."""
    return fitted(Camera(across, up, 0 * across, weight, 0.0, *limits(ax), 300), ax)


def offset_panel(ax: Axes, across: Extensor, up: Extensor) -> Camera:
    """The camera of a matplotlib panel reading its points by the planes `across` and `up`, each point an offset from
    the origin of weight one, as a vector algebra's points are; its limits the readings it spans, as many pixels across
    as the panel's room holds."""
    return fitted(Camera(across, up, 0 * across, 0 * across, 1.0, *limits(ax), 300), ax)


def overlay(ax: Axes, pictures: list, *, zorder: float | None = None):
    """Pictures of a panel, as its camera draws them, laid over each other in order and pinned into the panel at its
    own pixels, clear where none covers; the image, for an animation to set its data."""
    shown = pinned(ax, composited(pictures), (*ax.get_xlim(), *ax.get_ylim()))
    if zorder is not None:
        shown.set_zorder(zorder)
    return shown

