"""The atoms and the brightness of the scattered light in a section of momentum transfers, read out for matplotlib.

The drawing reads its values by pairing, the same lines for the crystal in spacetime and in the algebra of space.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.colors import LogNorm
from matplotlib.image import AxesImage

from numga import Extensor, stack
from examples.util import charts
from examples.util.animation import capture
from examples.util.wgsl import pinned

position_limit = 7.8
brightness_floor = 1e-4
# The room between panels and around them, in pixels.
gap = 6
# The atoms' and the rings' sizes, in square points, and the rings' width, thin so as not to hide the peaks they ring;
# the animation's panels, about twice as many pixels across, draw them half as large again.
atom_size = 5
ring_size = 50
ring_width = charts.points(1.0)
movie_scale = 1.5


# --- plumbing -------------------------------------------------------------------------
def places(points: Extensor) -> np.ndarray:
    """Points `[...]` as coordinates `[..., 2]` along x and y: their pairings with the planes x and y."""
    mv = points.context.multivector
    return (stack((mv.x, mv.y)) & points[..., None]).to_array()


def readings(wavevectors: Extensor) -> np.ndarray:
    """Wavevectors `[...]` as coordinates `[..., 2]` along x and y: their pairings with the points one along x and
    along y."""
    mv = wavevectors.context.multivector
    return (wavevectors[..., None] & stack((mv.x.dual(), mv.y.dual()))).to_array()


def panels(rows: int, columns: int, pixels: int) -> tuple[plt.Figure, list[list[Axes]]]:
    """A figure of square panels, each `pixels` across, on whole pixels and a gap apart."""
    span = pixels + gap
    width, height = columns * span + gap, rows * span + gap
    figure = plt.figure(figsize=(width / charts.DPI, height / charts.DPI), dpi=charts.DPI)
    axes = [[figure.add_axes(((gap + column * span) / width, (gap + (rows - 1 - row) * span) / height,
                              pixels / width, pixels / height))
             for column in range(columns)] for row in range(rows)]
    return figure, axes


def cross(ax: Axes) -> None:
    """A faint cross through the origin, its lines named x and y at their ends, framing the atoms."""
    ax.axhline(0, color="#cbd5e1", linewidth=charts.FRAME_WIDTH, zorder=0)
    ax.axvline(0, color="#cbd5e1", linewidth=charts.FRAME_WIDTH, zorder=0)
    ax.text(position_limit, 0, "$x$", ha="right", va="bottom", color="#94a3b8")
    ax.text(0, position_limit, " $y$", ha="left", va="top", color="#94a3b8")
    ax.set(xlim=(-position_limit, position_limit), ylim=(-position_limit, position_limit))
    ax.set_axis_off()


def atoms(ax: Axes, positions: Extensor) -> PathCollection:
    points = places(positions)                                          # [atoms, 2]
    dots = ax.scatter(points[:, 0], points[:, 1], s=atom_size, color="#0d9488", edgecolors="white", linewidths=0.2)
    cross(ax)
    return dots


def diffraction(ax: Axes, reciprocal: Extensor, transfers: Extensor,
                brightness: Extensor) -> tuple[AxesImage, PathCollection]:
    samples = readings(transfers)                                       # [rows, columns, 2]
    peaks = readings(reciprocal)                                        # [reflections, 2]
    # Pixel centres coincide with the sampled momentum transfers.
    pixel = samples[0, 1, 0] - samples[0, 0, 0]
    extent = (samples[0, 0, 0] - pixel / 2, samples[0, -1, 0] + pixel / 2,
              samples[0, 0, 1] - pixel / 2, samples[-1, 0, 1] + pixel / 2)
    # Rows run up in y; the image's first row is its top.
    image = pinned(ax, brightness.to_array()[::-1], extent)
    image.set(cmap="magma", norm=LogNorm(vmin=brightness_floor, vmax=1))
    rings = ax.scatter(peaks[:, 0], peaks[:, 1], facecolors="none", edgecolors="#67e8f9", s=ring_size,
                       linewidths=ring_width)
    ax.set(xlim=extent[:2], ylim=extent[2:])
    return image, rings


@charts.style()
def comparison(positions: Extensor, reciprocal: Extensor, transfers: Extensor, brightness: Extensor) -> plt.Figure:
    """For each case `[cases]`, its atoms above and the brightness of its diffraction pattern below, with the carried
    reciprocal lattice ringed."""
    n_cases = positions.shape[0]
    figure, (top, bottom) = panels(2, n_cases, transfers.shape[-1])
    for i in range(n_cases):
        atoms(top[i], positions[i])
        diffraction(bottom[i], reciprocal[i], transfers, brightness[i])
    return figure


def animate(frames: Iterable[tuple[Extensor, Extensor, Extensor]], transfers: Extensor) -> Iterator[np.ndarray]:
    """Per frame, the atoms beside the brightness of their diffraction pattern: the first frame drawn, and each one
    after it moving the atoms, the pattern and the rings."""
    frames = iter(frames)
    # The charts' style held while the frames are drawn, for the artists that drawing makes.
    with charts.style():
        figure, ((left, right),) = panels(1, 2, transfers.shape[-1])
        positions, reciprocal, brightness = next(frames)
        dots = atoms(left, positions)
        image, rings = diffraction(right, reciprocal, transfers, brightness)
        dots.set_sizes([movie_scale**2 * atom_size])
        rings.set_sizes([movie_scale**2 * ring_size])
        rings.set_linewidths(movie_scale * ring_width)
        yield capture(figure)
        for positions, reciprocal, brightness in frames:
            dots.set_offsets(places(positions))
            image.set_data(brightness.to_array()[::-1])
            rings.set_offsets(readings(reciprocal))
            yield capture(figure)
        plt.close(figure)
