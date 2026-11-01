"""The atoms and the brightness of the scattered light in a section of momentum transfers.

The atoms, read by the planes x and y of their algebra, and the rings around the reciprocal lattice's wavevectors,
read by the points one along x and along y, are drawn on the GPU as they are; the brightness is read out for
matplotlib by pairing. The same lines serve the crystal in spacetime and in the algebra of space.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm, to_rgb
from matplotlib.image import AxesImage

from numga import Extensor, stack
from examples.util import charts
from examples.util.animation import capture
from examples.util.wgsl import pinned, simplices
from examples.util.wgsl.camera import Camera, panel, read

position_limit = 7.8
brightness_floor = 1e-4
# The room between panels and around them, in pixels.
gap = 6
# The atoms' radius in pixels and their colour; the rings' radius in pixels, their width, thin so as not to hide the
# peaks they ring, their colour and the strokes around each; the faint cross's colour; the animation's panels, about
# twice as many pixels across, draw them half as large again.
atom_radius = 1.6
atom_colour = to_rgb("#0d9488")
ring_radius, ring_width = 4.9, 1.4
ring_colour = to_rgb("#67e8f9")
n_around = 24
cross_colour = to_rgb("#cbd5e1")
movie_scale = 1.5


# --- plumbing -------------------------------------------------------------------------
def panels(rows: int, columns: int, pixels: int) -> tuple[plt.Figure, list[list[Axes]]]:
    """A figure of square panels, each `pixels` across, on whole pixels and a gap apart."""
    span = pixels + gap
    width, height = columns * span + gap, rows * span + gap
    figure = plt.figure(figsize=(width / charts.DPI, height / charts.DPI), dpi=charts.DPI)
    axes = [[figure.add_axes(((gap + column * span) / width, (gap + (rows - 1 - row) * span) / height,
                              pixels / width, pixels / height))
             for column in range(columns)] for row in range(rows)]
    return figure, axes


def framing(ax: Axes, positions: Extensor) -> Camera:
    """The atoms' panel, the lines of its cross through the origin named x and y at their ends; its camera, reading the
    atoms by the planes x and y of their algebra."""
    ax.text(position_limit, 0, "$x$", ha="right", va="bottom", color="#94a3b8")
    ax.text(0, position_limit, " $y$", ha="left", va="top", color="#94a3b8")
    ax.set(xlim=(-position_limit, position_limit), ylim=(-position_limit, position_limit))
    ax.set_axis_off()
    mv = positions.context.multivector
    return panel(ax, mv.x, mv.y)


def atoms(positions: Extensor, radius: float, camera: Camera) -> np.ndarray:
    """The atoms `[atoms]` as dots `radius` pixels across, over a faint cross through the origin across the panel,
    along the directions the camera reads as one across and one up, the duals of its planes over their own readings:
    their picture."""
    along, rise = (plane.dual() / (plane & plane.dual()) for plane in (camera.across, camera.up))
    ends = position_limit * stack((-along, along, -rise, rise))         # [4] Antivector-
    return simplices.composited([
        simplices.strokes(ends, simplices.Topology.chains([2, 2]), cross_colour, charts.FRAME_PX, camera,
                          cap=simplices.Cap.BUTT),
        simplices.markers(positions, atom_colour, radius, camera)])


def rings(reciprocal: Extensor, scale: float, camera: Camera) -> np.ndarray:
    """Rings around the reciprocal lattice's wavevectors `[reflections]`, read as they are by a camera of the points one
    along x and along y, `scale` times their size, each a closed chain of strokes: their picture."""
    # The steps the camera reads as one across and one up, the duals of its planes over their own readings; a ring's
    # radius from each wavevector, the first step turned about their plane all the way round.
    along, rise = (plane.dual() / read(plane, plane.dual()) for plane in (camera.across, camera.up))
    turns = ((along ^ rise) * (-np.linspace(0, 2 * np.pi, n_around, endpoint=False) / 2)).exp()   # [around] Rotor+
    circle = (turns >> along) * (scale * ring_radius * camera.pixel)                              # [around] Vector+
    ends = (reciprocal[:, None] + circle).reshape(-1)                                             # [reflections * around] Vector+
    return simplices.strokes(ends, simplices.Topology.loops([n_around] * len(reciprocal)), ring_colour,
                             scale * ring_width, camera)


def diffraction(ax: Axes, reciprocal: Extensor, transfers: Extensor, brightness: Extensor,
                scale: float) -> tuple[AxesImage, AxesImage, Camera]:
    """The brightness over the section of momentum transfers, with the reciprocal lattice ringed `scale` times the
    rings' size: the brightness's image and the rings', for an animation to set their data, and the panel's camera."""
    # The sampled momentum transfers `[rows, columns]` read by the points one along x and along y; pixel centres
    # coincide with them, so the picture reaches half a step between samples past the outermost.
    mv = transfers.context.multivector
    readings = transfers.reshape(-1) & stack((mv.x.dual(), mv.y.dual()))[:, None]   # [2, samples] Scalar+
    low, high = readings.min(axis=-1), readings.max(axis=-1)                   # [2] Scalar+, [2] Scalar+
    half_step = (high - low) / (2 * (np.array(transfers.shape[::-1]) - 1))     # [2] Scalar+
    extent = tuple(stack((low - half_step, high + half_step), axis=-1).reshape(-1).to_array())
    # Rows run up in y; the image's first row is its top.
    image = pinned(ax, brightness.to_array()[::-1], extent)
    image.set(cmap="magma", norm=LogNorm(vmin=brightness_floor, vmax=1))
    ax.set(xlim=extent[:2], ylim=extent[2:])
    # The wavevectors read by the points one along x and along y, as the brightness's samples are.
    camera = panel(ax, mv.x.dual(), mv.y.dual())
    return image, simplices.overlay(ax, [rings(reciprocal, scale, camera)]), camera


@charts.style()
def comparison(positions: Extensor, reciprocal: Extensor, transfers: Extensor, brightness: Extensor) -> plt.Figure:
    """For each case `[cases]`, its atoms above and the brightness of its diffraction pattern below, with the carried
    reciprocal lattice ringed."""
    n_cases = positions.shape[0]
    figure, (top, bottom) = panels(2, n_cases, transfers.shape[-1])
    for i in range(n_cases):
        simplices.overlay(top[i], [atoms(positions[i], atom_radius, framing(top[i], positions[i]))])
        diffraction(bottom[i], reciprocal[i], transfers, brightness[i], 1.0)
    return figure


def animate(frames: Iterable[tuple[Extensor, Extensor, Extensor]], transfers: Extensor) -> Iterator[np.ndarray]:
    """Per frame, the atoms beside the brightness of their diffraction pattern: the first frame drawn, and each one
    after it moving the atoms, the pattern and the rings."""
    frames = iter(frames)
    # The charts' style held while the frames are drawn, for the artists that drawing makes.
    with charts.style():
        figure, ((left, right),) = panels(1, 2, transfers.shape[-1])
        positions, reciprocal, brightness = next(frames)
        atom_camera = framing(left, positions)
        atoms_seen = simplices.overlay(left, [atoms(positions, movie_scale * atom_radius, atom_camera)])
        image, rings_seen, ring_camera = diffraction(right, reciprocal, transfers, brightness, movie_scale)
        yield capture(figure)
        for positions, reciprocal, brightness in frames:
            atoms_seen.set_data(atoms(positions, movie_scale * atom_radius, atom_camera))
            image.set_data(brightness.to_array()[::-1])
            rings_seen.set_data(rings(reciprocal, movie_scale, ring_camera))
            yield capture(figure)
        plt.close(figure)
