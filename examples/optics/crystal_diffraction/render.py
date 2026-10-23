"""The atoms and the coherent intensity in a reciprocal-space section."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.image import AxesImage
from matplotlib.colors import LogNorm

from numga import NumpyContext, stack
from numga.algebras import STA
from examples.util.animation import capture
from examples.optics.crystal_diffraction import core

POSITION_LIMIT = 7.8
INTENSITY_FLOOR = 1e-4


# --- plumbing -------------------------------------------------------------------------
def cross(ax: Axes) -> None:
    """A faint cross through the origin, its lines named x and y at their ends, framing the atoms."""
    ax.axhline(0, color="#cbd5e1", linewidth=0.7, zorder=0)
    ax.axvline(0, color="#cbd5e1", linewidth=0.7, zorder=0)
    ax.text(POSITION_LIMIT, 0, "$x$", ha="right", va="bottom", color="#94a3b8", fontsize=11)
    ax.text(0, POSITION_LIMIT, " $y$", ha="left", va="top", color="#94a3b8", fontsize=11)
    ax.set(xlim=(-POSITION_LIMIT, POSITION_LIMIT), ylim=(-POSITION_LIMIT, POSITION_LIMIT), xticks=[], yticks=[])
    ax.set_aspect("equal")
    ax.spines[:].set_visible(False)


def atoms(ax: Axes, positions: core.Point) -> PathCollection:
    points = positions.cast(core.ga.subspace("yz zx xy")).kernel
    dots = ax.scatter(points[:, 0], points[:, 1], s=22, color="#0d9488", edgecolors="white", linewidths=0.4)
    cross(ax)
    return dots


def diffraction(ax: Axes, reciprocal: core.Plane, grid: core.Plane,
                intensity: core.Scalar) -> tuple[AxesImage, PathCollection]:
    samples = grid.cast(core.ga.subspace("x y z")).kernel
    peaks = reciprocal.cast(core.ga.subspace("x y z")).kernel
    values = intensity.kernel[..., 0]
    # Pixel centres coincide with the sampled momentum transfers.
    pixel = samples[0, 1, 0] - samples[0, 0, 0]
    extent = (samples[0, 0, 0] - pixel / 2, samples[0, -1, 0] + pixel / 2,
              samples[0, 0, 1] - pixel / 2, samples[-1, 0, 1] + pixel / 2)
    image = ax.imshow(values, origin="lower", extent=extent, cmap="magma", interpolation="nearest",
                      norm=LogNorm(vmin=INTENSITY_FLOOR, vmax=1))
    rings = ax.scatter(peaks[:, 0], peaks[:, 1], facecolors="none", edgecolors="#67e8f9", s=62, linewidths=0.9)
    ax.set(xlim=extent[:2], ylim=extent[2:], xticks=[], yticks=[])
    return image, rings


def comparison(positions: core.Point, reciprocal: core.Plane, grid: core.Plane, intensity: core.Scalar) -> plt.Figure:
    """For each case `[cases]`, its atoms above and its diffraction pattern below, with the carried reciprocal
    lattice ringed."""
    figure = plt.figure(figsize=(12, 8), layout="constrained")
    count = positions.shape[0]
    for i in range(count):
        atoms(figure.add_subplot(2, count, i + 1), positions[i])
        diffraction(figure.add_subplot(2, count, count + i + 1), reciprocal[i], grid, intensity[i])
    return figure


def animate(frames: Iterable[tuple[core.Point, core.Plane, core.Scalar]], grid: core.Plane) -> Iterator[np.ndarray]:
    """Per frame, the atoms beside their diffraction pattern: the first frame drawn, and each one after it moving the
    atoms, the pattern and the rings."""
    frames = iter(frames)
    figure, panels = plt.subplots(1, 2, figsize=(6, 3), dpi=70, layout="constrained")
    positions, reciprocal, intensity = next(frames)
    dots = atoms(panels[0], positions)
    image, rings = diffraction(panels[1], reciprocal, grid, intensity)
    yield capture(figure)
    for positions, reciprocal, intensity in frames:
        dots.set_offsets(positions.cast(core.ga.subspace("yz zx xy")).kernel[:, :2])
        image.set_data(intensity.kernel[..., 0])
        rings.set_offsets(reciprocal.cast(core.ga.subspace("x y z")).kernel[:, :2])
        yield capture(figure)
    plt.close(figure)


# --- spacetime read-out ---------------------------------------------------------------
# Events, wavevectors and transfers in the algebra of spacetime, read along the observer's directions in space.
SpacetimeVector = STA.gatype.vector()
# Events are antivectors of spacetime.
SpacetimeAntivector = STA.gatype.antivector()
SPACETIME = NumpyContext(STA).multivector
SpacetimeScalar = STA.gatype.scalar()
OBSERVER_SPACE = STA.subspace("x y z")


def _draw_atoms(ax: Axes, points: np.ndarray) -> None:
    ax.scatter(points[:, 0], points[:, 1], s=22, color="#0d9488", edgecolors="white", linewidths=0.4)
    cross(ax)


def _draw_intensity(ax: Axes, peaks: np.ndarray, samples: np.ndarray, values: np.ndarray) -> None:
    # Pixel centres coincide with the sampled momentum transfers.
    pixel = samples[0, 1, 0] - samples[0, 0, 0]
    extent = (samples[0, 0, 0] - pixel / 2, samples[0, -1, 0] + pixel / 2,
              samples[0, 0, 1] - pixel / 2, samples[-1, 0, 1] + pixel / 2)
    ax.imshow(values, origin="lower", extent=extent, cmap="magma", interpolation="nearest",
              norm=LogNorm(vmin=INTENSITY_FLOOR, vmax=1))
    ax.scatter(peaks[:, 0], peaks[:, 1], facecolors="none", edgecolors="#67e8f9", s=62, linewidths=0.9)
    ax.set(xlim=extent[:2], ylim=extent[2:], xticks=[], yticks=[])


def spacetime_comparison(positions: SpacetimeAntivector, reciprocal: SpacetimeVector, transfers: SpacetimeVector,
                         energy_density: SpacetimeScalar) -> plt.Figure:
    """The atoms above and the energy density of the scattered field below, one column per case."""
    # The events' positions: their pairings with the planes x, y and z.
    points = (stack((SPACETIME.x, SPACETIME.y, SPACETIME.z)) & positions[..., None]).to_array()   # [cases, atoms, 3]
    peaks = reciprocal.cast(OBSERVER_SPACE).kernel                      # [cases, reflections, 3]
    samples = transfers.cast(OBSERVER_SPACE).kernel                     # [rows, columns, 3]
    values = energy_density.kernel[..., 0]                              # [cases, rows, columns]
    n_cases = points.shape[0]
    figure = plt.figure(figsize=(12, 8), layout="constrained")
    for i in range(n_cases):
        _draw_atoms(figure.add_subplot(2, n_cases, i + 1), points[i])
        _draw_intensity(figure.add_subplot(2, n_cases, n_cases + i + 1), peaks[i], samples, values[i])
    return figure
