"""Drawing for the Dirac electron: the mass shell, and trembling paths with their spin."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.animation import capture
from examples.relativity.dirac import core

COLOURS = ("#2e86c1", "#7d3c98", "#c0392b")


def relative(bivectors: core.Bivector) -> np.ndarray:
    """Components of relative vectors along the bivectors xt, yt and zt, which square to +1."""
    return bivectors.cast(core.ga.subspace("xt yt zt")).kernel


def spatial(vectors: core.Vector) -> np.ndarray:
    """The spatial components of spacetime vectors, as the observer sees them: their products with
    the time axis, read as relative vectors."""
    return relative(vectors ^ core.TIME)


def draw_mass_shell(momenta: core.Vector, values: core.Scalar, mass: float) -> plt.Figure:
    """The positive- and negative-energy sheets over the momentum plane, with the gap of twice
    the mass between them."""
    p = spatial(momenta)                                                       # [n, n, 3]
    energies = values.to_array()                                               # [n, n, 8]
    figure = plt.figure(figsize=(7, 6))
    ax = figure.add_subplot(projection="3d")
    for sheet, colour in ((energies[..., -1], "#c0392b"), (energies[..., 0], "#2e86c1")):
        ax.plot_surface(p[..., 0], p[..., 1], sheet, color=colour, alpha=0.55, linewidth=0, antialiased=True)
    ax.set_xlabel("momentum x (mc)")
    ax.set_ylabel("momentum y (mc)")
    ax.set_zlabel("energy (mc²)")
    ax.set_title(f"the mass shell: energies ±√(p² + m²), gap {2 * mass:g} mc²")
    ax.view_init(elev=14, azim=-58)
    figure.tight_layout()
    return figure


def draw_paths(times: np.ndarray, spinors: list, paths: list, mixtures: tuple) -> plt.Figure:
    """The paths the currents trace, one per share of negative energy, with the spin at the start."""
    figure = plt.figure(figsize=(7, 7))
    ax = figure.add_subplot(projection="3d")
    draw_scene(ax, spinors, paths, mixtures, len(times))
    figure.tight_layout()
    return figure


def draw_scene(ax, spinors: list, paths: list, mixtures: tuple, upto: int) -> None:
    for psi, trace, share, colour in zip(spinors, paths, mixtures, COLOURS):
        xyz = relative(trace)[:upto]                                           # [times, 3]
        ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=colour, linewidth=1.2, label=f"{share:.0%} negative energy")
        here = xyz[-1]
        axis = spatial(core.spin(psi[upto - 1]))
        axis = 0.4 * axis / np.linalg.norm(axis)
        ax.quiver(*here, *axis, color=colour, linewidth=1.5, arrow_length_ratio=0.25)
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(-0.6, 0.6)
    ax.set_zlim(-0.2, 3.2)
    ax.set_box_aspect((1.2, 1.2, 3.4))
    ax.set_xlabel("x (ħ/mc)")
    ax.set_ylabel("y (ħ/mc)")
    ax.set_zlabel("z (ħ/mc)")
    ax.legend(loc="upper left", fontsize=8)
    ax.view_init(elev=12, azim=-60)


def animate_paths(times: np.ndarray, spinors: list, paths: list, mixtures: tuple, every: int) -> list[np.ndarray]:
    """The paths drawn as they are traced, with each electron's spin at its current position."""
    frames = []
    for upto in range(every, len(times) + 1, every):
        figure = plt.figure(figsize=(5, 6))
        ax = figure.add_subplot(projection="3d")
        draw_scene(ax, spinors, paths, mixtures, upto)
        ax.set_title(f"t = {times[upto - 1]:.1f} ħ/mc²")
        figure.tight_layout()
        frames.append(capture(figure))
        plt.close(figure)
    return frames
