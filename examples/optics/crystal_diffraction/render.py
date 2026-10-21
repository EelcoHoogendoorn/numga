"""The atoms and the coherent intensity in a reciprocal-space section."""

from __future__ import annotations

from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm

from numga import NumpyContext, stack
from numga.algebras import STA
from examples.animation import capture
from examples.optics.crystal_diffraction import core

POSITION_LIMIT = 7.8
INTENSITY_FLOOR = 1e-4


# --- plumbing -------------------------------------------------------------------------
def atoms(ax: Axes, positions: core.Point) -> None:
    points = positions.cast(core.ga.subspace("yz zx xy")).kernel
    ax.scatter(points[:, 0], points[:, 1], s=22, color="#0d9488", edgecolors="white", linewidths=0.4)
    ax.axhline(0, color="#cbd5e1", linewidth=0.7, zorder=0)
    ax.axvline(0, color="#cbd5e1", linewidth=0.7, zorder=0)
    ax.set(xlim=(-POSITION_LIMIT, POSITION_LIMIT), ylim=(-POSITION_LIMIT, POSITION_LIMIT), xticks=[], yticks=[])
    ax.set_aspect("equal")
    ax.spines[:].set_visible(False)


def diffraction(ax: Axes, reciprocal: core.Plane, grid: core.Plane, intensity: core.Scalar) -> None:
    samples = grid.cast(core.ga.subspace("x y z")).kernel
    peaks = reciprocal.cast(core.ga.subspace("x y z")).kernel
    values = intensity.kernel[..., 0]
    # Pixel centres coincide with the sampled momentum transfers.
    pixel = samples[0, 1, 0] - samples[0, 0, 0]
    extent = (samples[0, 0, 0] - pixel / 2, samples[0, -1, 0] + pixel / 2,
              samples[0, 0, 1] - pixel / 2, samples[-1, 0, 1] + pixel / 2)
    ax.imshow(values, origin="lower", extent=extent, cmap="magma", interpolation="nearest",
              norm=LogNorm(vmin=INTENSITY_FLOOR, vmax=1))
    ax.scatter(peaks[:, 0], peaks[:, 1], facecolors="none", edgecolors="#67e8f9", s=62, linewidths=0.9)
    ax.set(xlim=extent[:2], ylim=extent[2:], xticks=[], yticks=[])


def comparison(positions: core.Point, reciprocal: core.Plane, grid: core.Plane, intensity: core.Scalar) -> plt.Figure:
    """For each case `[cases]`, its atoms above and its diffraction pattern below, with the carried reciprocal
    lattice ringed."""
    figure = plt.figure(figsize=(12, 8), layout="constrained")
    count = positions.shape[0]
    for i in range(count):
        atoms(figure.add_subplot(2, count, i + 1), positions[i])
        diffraction(figure.add_subplot(2, count, count + i + 1), reciprocal[i], grid, intensity[i])
    return figure


def frame(positions: core.Point, reciprocal: core.Plane, intensity: core.Scalar, grid: core.Plane) -> np.ndarray:
    figure, panels = plt.subplots(1, 2, figsize=(6, 3), dpi=70, layout="constrained")
    atoms(panels[0], positions)
    diffraction(panels[1], reciprocal, grid, intensity)
    pixels = capture(figure)
    plt.close(figure)
    return pixels


def animate(frames: Iterable[tuple[core.Point, core.Plane, core.Scalar]], grid: core.Plane) -> list[np.ndarray]:
    return [frame(positions, reciprocal, intensity, grid) for positions, reciprocal, intensity in frames]


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
    ax.axhline(0, color="#cbd5e1", linewidth=0.7, zorder=0)
    ax.axvline(0, color="#cbd5e1", linewidth=0.7, zorder=0)
    ax.set(xlim=(-POSITION_LIMIT, POSITION_LIMIT), ylim=(-POSITION_LIMIT, POSITION_LIMIT), xticks=[], yticks=[])
    ax.set_aspect("equal")
    ax.spines[:].set_visible(False)


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
