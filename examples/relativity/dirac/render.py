"""Drawing for the Dirac electron: the mass shell, and trembling paths with their spin, drawn on the GPU."""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D

from numga import concatenate, stack
from examples.util import charts
from examples.util.animation import capture
from examples.relativity.dirac import core
from examples.util.wgsl import fitted, show, simplices, solids

# The observer sees the point where its time line meets the spatial slice.
space_slice = core.mv.t ^ (core.Antivector & core.time)

colours = ("#2e86c1", "#7d3c98", "#c0392b")
BACKGROUND = np.ones(3)
# The view's elevation and azimuth in radians, azimuth zero looking along +y, and the margin around the paths as a
# share of the larger side.
ELEVATION, AZIMUTH = np.radians(12.0), np.radians(30.0)
MARGIN = 0.1
# The spin arrows' length, in the paths' units; the widths of the paths and the arrows and the radius of the balls at
# the arrows' tips, in pixels.
SPIN_LENGTH = 0.4
PATH_WIDTH, ARROW_WIDTH, TIP_RADIUS = 1.5, 1.5, 2.5


def spin_arrow(psi: core.Spinor) -> core.Antivector:
    """The spin's axis as the observer sees it, drawn as an arrow of the given length: the line where the spin's
    patch crosses the observer's slice of space, at unit length, which a direction within the slice squares to."""
    axis = core.spin(psi) ^ core.mv.t                                           # [...] Antivector-
    return SPIN_LENGTH * axis / axis.squared().square_root()                    # [...] Antivector-


def framing(ax, places: list, tips: list) -> solids.View:
    """One view for every frame, framing the whole of each path `[mixtures] Point-[times]` and every spin arrow's tip
    along it, as many pixels across as the axes' room holds."""
    return fitted(solids.view([*places, *tips], 0.0, ELEVATION, AZIMUTH, MARGIN, 1), ax)


def picture(places: list, tips: list, upto: int, view: solids.View) -> np.ndarray:
    """The paths `[mixtures] Point-[times]` traced up to a time, each with its spin as an arrow from its end to the arrow's
    tip, `[rows, columns, 3]`."""
    rgb = np.array([to_rgb(colour) for colour in colours[: len(places)]])
    starts = stack([place[upto - 1] for place in places])                       # [mixtures] Point-
    ends = stack([tip[upto - 1] for tip in tips])                               # [mixtures] Point-
    # Each path up to the time a chain of edges, each edge in its path's colour.
    trails = concatenate([place[:upto] for place in places])                    # [mixtures * upto] Point-
    lines = simplices.tubes(trails, simplices.Mesh.chains([upto] * len(places)), np.repeat(rgb, upto - 1, axis=0),
                            PATH_WIDTH, view, solids.gloss)
    # Each spin's arrow a rod with a ball at its tip.
    shafts = solids.rods(starts, ends, ARROW_WIDTH / 2 * view.pixel)
    heads = solids.ball(ends, TIP_RADIUS * view.pixel)[:, None]
    return solids.draw([(shafts, rgb), (heads, rgb)], view, BACKGROUND, inks=[], translucent=[], meshes=[(lines, 1.0)])


def draw_scene(ax, places: list, tips: list, mixtures: tuple, upto: int, view: solids.View) -> AxesImage:
    """The paths traced up to a time, each with its spin as an arrow at its end, and the legend naming them. The
    picture shown, for an animation to set its data."""
    shown = show(ax, picture(places, tips, upto, view), view)
    # The tall image in the middle of the figure, the legend in the figure's corner beside it.
    ax.figure.legend(handles=[Line2D([], [], color=colour, label=f"{share:.0%} negative energy")
                              for share, colour in zip(mixtures, colours)], loc="upper left")
    return shown


@charts.style()
def draw_paths(times: np.ndarray, spinors: list, paths: list, mixtures: tuple) -> plt.Figure:
    """The paths the currents trace, one per share of negative energy, with the spin at the end."""
    # The projective embedding of each path as the observer slices it, and of its spin arrows' tips, the sliced
    # arrows directions at infinity.
    places = [solids.origin + solids.context.include_from_algebra(space_slice(trace))
              for trace in paths]                                               # [mixtures] Point-[times]
    tips = [place + solids.context.include_from_algebra(space_slice(spin_arrow(psi)))
            for place, psi in zip(places, spinors)]                             # [mixtures] Point-[times]
    figure = plt.figure(figsize=(440 / charts.DPI, 350 / charts.DPI))
    ax = figure.add_axes((0, 0, 1, 1))
    draw_scene(ax, places, tips, mixtures, len(times), framing(ax, places, tips))
    return figure


def animate_paths(times: np.ndarray, spinors: list, paths: list, mixtures: tuple, every: int) -> Iterator[np.ndarray]:
    """The paths drawn as they are traced, with each electron's spin at its current position."""
    # The projective embedding of each path as the observer slices it, and of its spin arrows' tips, the sliced
    # arrows directions at infinity.
    places = [solids.origin + solids.context.include_from_algebra(space_slice(trace))
              for trace in paths]                                               # [mixtures] Point-[times]
    tips = [place + solids.context.include_from_algebra(space_slice(spin_arrow(psi)))
            for place, psi in zip(places, spinors)]                             # [mixtures] Point-[times]
    with charts.style():
        figure = plt.figure(figsize=(420 / charts.DPI, 300 / charts.DPI))
        ax = figure.add_axes((0, 0, 1, 1))
        view = framing(ax, places, tips)
        # The figure and its legend set up once; only the picture changes between frames.
        shown = draw_scene(ax, places, tips, mixtures, every, view)
        yield capture(figure)
        for upto in range(2 * every, len(times) + 1, every):
            shown.set_data(picture(places, tips, upto, view))
            yield capture(figure)
        plt.close(figure)
