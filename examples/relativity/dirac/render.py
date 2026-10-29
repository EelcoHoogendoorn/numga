"""Drawing for the Dirac electron: the mass shell, and trembling paths with their spin, drawn on the GPU."""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D

from numga import stack
from examples.util import charts
from examples.util.animation import capture
from examples.relativity.dirac import core
from examples.util.wgsl import fitted, show, solids, strokes

sm = solids.mv
colours = ("#2e86c1", "#7d3c98", "#c0392b")
BACKGROUND = np.ones(3)
# The view's elevation and azimuth in radians, azimuth zero looking along +y, and the margin around the paths as a
# share of the larger side.
ELEVATION, AZIMUTH = np.radians(12.0), np.radians(30.0)
MARGIN = 0.1
# The spin arrows' length, in the paths' units; the widths of the paths and the arrows and the arrowheads' length and
# width, in pixels.
SPIN_LENGTH = 0.4
PATH_WIDTH, ARROW_WIDTH, HEAD_LENGTH, HEAD_WIDTH = 1.5, 1.5, 9.0, 5.0


def spatial(directions: core.Antivector) -> np.ndarray:
    """Directions as the observer sees them: read by pairing with the slices x, y and z, on the last axis."""
    mv = directions.context.multivector
    return (stack((mv.x, mv.y, mv.z)) & directions[..., None]).to_array()


def framing(ax, spinors: list, paths: list) -> solids.View:
    """One view for every frame, framing the whole of each path and every spin arrow along it, as many pixels across
    as the axes' room holds."""
    ends = [spatial(trace) + SPIN_LENGTH * unit(spatial(axis(psi))) for psi, trace in zip(spinors, paths)]
    return fitted(solids.view([sm.point(spatial(trace)) for trace in paths] + [sm.point(tips) for tips in ends], 0.0,
                              ELEVATION, AZIMUTH, MARGIN, 1), ax)


def axis(psi: core.Spinor) -> core.Antivector:
    """The spin's axis as the observer sees it, drawn as an arrow: the line where the spin's patch crosses the
    observer's slice of space."""
    return core.spin(psi) ^ core.mv.t


def unit(directions: np.ndarray) -> np.ndarray:
    return directions / np.linalg.norm(directions, axis=-1, keepdims=True)


def picture(spinors: list, paths: list, upto: int, view: solids.View) -> np.ndarray:
    """The paths traced up to a time, each with its spin as an arrow at its end, `[rows, columns, 3]`."""
    rgb = np.array([to_rgb(colour) for colour in colours[: len(paths)]])
    traced = [spatial(trace)[:upto] for trace in paths]                         # [times, 3] each
    here = np.stack([xyz[-1] for xyz in traced])                                # [mixtures, 3]
    spins = unit(np.stack([spatial(axis(psi[upto - 1])) for psi in spinors]))   # [mixtures, 3]
    lines = strokes.polylines([(sm.point(xyz), colour) for xyz, colour in zip(traced, rgb)], PATH_WIDTH, view)
    shafts, heads = solids.arrows(sm.point(here), sm.point(here + SPIN_LENGTH * spins), ARROW_WIDTH / 2 * view.pixel,
                                  HEAD_LENGTH * view.pixel, HEAD_WIDTH / 2 * view.pixel)
    return solids.draw([(shafts, rgb), (heads, rgb)], view, BACKGROUND, inks=[], translucent=[], meshes=[(lines, 1.0)])


def draw_scene(ax, spinors: list, paths: list, mixtures: tuple, upto: int, view: solids.View) -> AxesImage:
    """The paths traced up to a time, each with its spin as an arrow at its end, and the legend naming them. The
    picture shown, for an animation to set its data."""
    shown = show(ax, picture(spinors, paths, upto, view), view)
    # The tall image in the middle of the figure, the legend in the figure's corner beside it.
    ax.figure.legend(handles=[Line2D([], [], color=colour, label=f"{share:.0%} negative energy")
                              for share, colour in zip(mixtures, colours)], loc="upper left")
    return shown


@charts.style()
def draw_paths(times: np.ndarray, spinors: list, paths: list, mixtures: tuple) -> plt.Figure:
    """The paths the currents trace, one per share of negative energy, with the spin at the end."""
    figure = plt.figure(figsize=(440 / charts.DPI, 350 / charts.DPI))
    ax = figure.add_axes((0, 0, 1, 1))
    draw_scene(ax, spinors, paths, mixtures, len(times), framing(ax, spinors, paths))
    return figure


def animate_paths(times: np.ndarray, spinors: list, paths: list, mixtures: tuple, every: int) -> Iterator[np.ndarray]:
    """The paths drawn as they are traced, with each electron's spin at its current position."""
    with charts.style():
        figure = plt.figure(figsize=(420 / charts.DPI, 300 / charts.DPI))
        ax = figure.add_axes((0, 0, 1, 1))
        view = framing(ax, spinors, paths)
        # The figure and its legend set up once; only the picture changes between frames.
        shown = draw_scene(ax, spinors, paths, mixtures, every, view)
        yield capture(figure)
        for upto in range(2 * every, len(times) + 1, every):
            shown.set_data(picture(spinors, paths, upto, view))
            yield capture(figure)
        plt.close(figure)
