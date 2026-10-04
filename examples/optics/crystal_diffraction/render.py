"""The atoms and the coherent intensity in a reciprocal-space section."""

from __future__ import annotations

from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm

from examples.animation import capture
from examples.optics.crystal_diffraction import core

POSITION_LIMIT = 7.8
INTENSITY_FLOOR = 1e-4


# --- plumbing -------------------------------------------------------------------------
def atoms(ax: Axes, positions: core.Vector) -> None:
    points = positions.cast(core.ga.subspace("x y z")).kernel
    ax.scatter(points[:, 0], points[:, 1], s=22, color="#0d9488", edgecolors="white", linewidths=0.4)
    ax.axhline(0, color="#cbd5e1", linewidth=0.7, zorder=0)
    ax.axvline(0, color="#cbd5e1", linewidth=0.7, zorder=0)
    ax.set(xlim=(-POSITION_LIMIT, POSITION_LIMIT), ylim=(-POSITION_LIMIT, POSITION_LIMIT), xticks=[], yticks=[])
    ax.set_aspect("equal")
    ax.spines[:].set_visible(False)


def diffraction(ax: Axes, reciprocal: core.Vector, grid: core.Vector, intensity: core.Scalar) -> None:
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


def comparison(crystal: core.Crystal, grid: core.Vector, intensity: core.Scalar, names: tuple[str, ...]) -> plt.Figure:
    figure = plt.figure(figsize=(12, 8), layout="constrained")
    count = len(names)
    lower = [figure.add_subplot(2, count, count + i + 1) for i in range(count)]
    for i, name in enumerate(names):
        ax = figure.add_subplot(2, count, i + 1)
        atoms(ax, crystal.positions[i])
        ax.set_title(name, fontsize=11)
        diffraction(lower[i], crystal.reciprocal[i], grid, intensity[i])
    return figure


def frame(positions: core.Vector, reciprocal: core.Vector, intensity: core.Scalar, grid: core.Vector) -> np.ndarray:
    figure, panels = plt.subplots(1, 2, figsize=(6, 3), dpi=70, layout="constrained")
    atoms(panels[0], positions)
    diffraction(panels[1], reciprocal, grid, intensity)
    pixels = capture(figure)
    plt.close(figure)
    return pixels


def animate(frames: Iterable[tuple[core.Vector, core.Vector, core.Scalar]], grid: core.Vector) -> list[np.ndarray]:
    return [frame(positions, reciprocal, intensity, grid) for positions, reciprocal, intensity in frames]
