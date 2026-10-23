"""Drawing for the Dirac electron: the mass shell, and trembling paths with their spin, drawn on the GPU."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D

from numga import stack
from examples.util.animation import capture
from examples.relativity.dirac import core
from examples.util.wgsl import meshes, show, solids

sm = solids.mv
colours = ("#2e86c1", "#7d3c98", "#c0392b")
BACKGROUND = np.ones(3)
# The view's elevation and azimuth in radians, azimuth zero looking along +y, the margin around the paths as a share
# of the larger side, and the pixels up the still and up a frame.
ELEVATION, AZIMUTH = np.radians(12.0), np.radians(30.0)
MARGIN, STILL_PIXELS, FRAME_PIXELS = 0.1, 680, 580
# The spin arrows' length, in the paths' units; the points around a tube; the widths of the paths and the arrows and
# the arrowheads' length and width, in pixels.
SPIN_LENGTH = 0.4
N_AROUND = 8
PATH_WIDTH, ARROW_WIDTH, HEAD_LENGTH, HEAD_WIDTH = 3.0, 3.0, 18.0, 10.0


def spatial(directions: core.Antivector) -> np.ndarray:
    """Directions as the observer sees them: read by pairing with the slices x, y and z, on the last axis."""
    mv = directions.context.multivector
    return (stack((mv.x, mv.y, mv.z)) & directions[..., None]).to_array()


def framing(spinors: list, paths: list, rows: int) -> solids.View:
    """One view for every frame, framing the whole of each path and every spin arrow along it, `rows` pixels up."""
    ends = [spatial(trace) + SPIN_LENGTH * unit(spatial(core.spin(psi))) for psi, trace in zip(spinors, paths)]
    framed = solids.view([sm.point(spatial(trace)) for trace in paths] + [sm.point(tips) for tips in ends], 0.0,
                         ELEVATION, AZIMUTH, MARGIN, rows)
    return framed._replace(pixels=round(rows * framed.half[0] / framed.half[1]))


def unit(directions: np.ndarray) -> np.ndarray:
    return directions / np.linalg.norm(directions, axis=-1, keepdims=True)


def draw_scene(ax, spinors: list, paths: list, mixtures: tuple, upto: int, view: solids.View) -> None:
    """The paths traced up to a time, each with its spin as an arrow at its end, and the legend naming them."""
    rgb = np.array([to_rgb(colour) for colour in colours[: len(paths)]])
    traced = [spatial(trace)[:upto] for trace in paths]                         # [times, 3] each
    here = np.stack([xyz[-1] for xyz in traced])                                # [mixtures, 3]
    spins = unit(np.stack([spatial(core.spin(psi[upto - 1])) for psi in spinors]))   # [mixtures, 3]
    tubes = meshes.tube_samples(list(zip(traced, rgb)), PATH_WIDTH, view, N_AROUND)
    shafts, heads = solids.arrows(sm.point(here), sm.point(here + SPIN_LENGTH * spins), ARROW_WIDTH / 2 * view.pixel,
                                  HEAD_LENGTH * view.pixel, HEAD_WIDTH / 2 * view.pixel)
    show(ax, solids.draw([(shafts, rgb), (heads, rgb)], view, BACKGROUND, meshes=[(tubes, 1.0)]), view)
    # The tall image in the middle of the figure, the legend in the space beside it.
    figure_width, figure_height = ax.figure.get_size_inches()
    ax.set_xlim(-view.half[1] * figure_width / figure_height, view.half[1] * figure_width / figure_height)
    ax.legend(handles=[Line2D([], [], color=colour, label=f"{share:.0%} negative energy")
                       for share, colour in zip(mixtures, colours)], loc="upper left", fontsize=8)


def draw_paths(times: np.ndarray, spinors: list, paths: list, mixtures: tuple) -> plt.Figure:
    """The paths the currents trace, one per share of negative energy, with the spin at the end."""
    figure = plt.figure(figsize=(7, 7))
    draw_scene(figure.add_axes((0, 0, 1, 1)), spinors, paths, mixtures, len(times),
               framing(spinors, paths, STILL_PIXELS))
    return figure


def animate_paths(times: np.ndarray, spinors: list, paths: list, mixtures: tuple, every: int) -> list[np.ndarray]:
    """The paths drawn as they are traced, with each electron's spin at its current position."""
    view = framing(spinors, paths, FRAME_PIXELS)
    frames = []
    for upto in range(every, len(times) + 1, every):
        figure = plt.figure(figsize=(5, 6))
        draw_scene(figure.add_axes((0, 0, 1, 1)), spinors, paths, mixtures, upto, view)
        frames.append(capture(figure))
        plt.close(figure)
    return frames
