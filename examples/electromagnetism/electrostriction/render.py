"""The reference lattice, its deformed bonds and dipoles, and its loading curve."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from examples.animation import capture
from examples.electromagnetism.electrostriction.core import Scalar, Vector

REST = "#c4cbd0"
BOND = "#286580"
DIPOLE = "#df6435"
APPLIED = "#bd9224"
TRANSVERSE = "#8271a6"
MARGIN = 0.55
FIELD_GAP = 0.8


def coordinates(value: Vector) -> np.ndarray:
    return np.asarray(value.cast(value.algebra.subspace("x y")).kernel)


def lattice_axes(sites: np.ndarray) -> tuple[plt.Figure, plt.Axes, np.ndarray]:
    """A figure framing the reference sites, with room below them for the applied field."""
    figure, ax = plt.subplots(figsize=(6, 6), dpi=90)
    reach = np.abs(sites).max(axis=0) + MARGIN
    ax.set(xlim=(-reach[0], reach[0]), ylim=(-reach[1] - FIELD_GAP - 0.35, reach[1]), aspect="equal")
    ax.axis("off")
    figure.subplots_adjust(0.02, 0.03, 0.98, 0.98)
    return figure, ax, reach


def draw_rest(rest: Vector, bonds: np.ndarray) -> plt.Figure:
    """The reference lattice: its particles and the bonds between them."""
    sites = coordinates(rest)
    figure, ax, _ = lattice_axes(sites)
    ax.add_collection(LineCollection(sites[bonds], colors=BOND, linewidths=2, zorder=2))
    ax.scatter(*sites.T, s=34, color=BOND, zorder=4)
    return figure


class LatticeView:
    """Artists for a lattice at actual scale; the arrows have fixed scales across frames."""

    def __init__(self, rest: Vector, bonds: np.ndarray, dipole_scale: float, field_scale: float) -> None:
        self.bonds, self.dipole_scale, self.field_scale = bonds, dipole_scale, field_scale
        sites = coordinates(rest)
        self.figure, ax, reach = lattice_axes(sites)
        ax.add_collection(LineCollection(sites[bonds], colors=REST, linewidths=1.2,
                                        linestyles="dashed", zorder=0))
        ax.scatter(*sites.T, s=22, facecolors="white", edgecolors=REST, linewidths=1, zorder=1)
        self.links = LineCollection(sites[bonds], colors=BOND, linewidths=2, zorder=2)
        ax.add_collection(self.links)
        self.particles = ax.scatter(*sites.T, s=34, color=BOND, zorder=4)
        self.dipoles = ax.quiver(*sites.T, np.zeros(len(sites)), np.zeros(len(sites)),
                                 color=DIPOLE, angles="xy", scale_units="xy", scale=1,
                                 pivot="middle", width=0.009, headwidth=3, headlength=4, zorder=5)
        field_y = -reach[1] - FIELD_GAP / 2
        self.field = ax.quiver([0], [field_y], [0], [0], color=APPLIED, angles="xy",
                               scale_units="xy", scale=1, pivot="middle", width=0.012,
                               headwidth=3, headlength=4)

    def update(self, positions: Vector, dipoles: Vector, applied: Vector) -> plt.Figure:
        sites = coordinates(positions)
        self.links.set_segments(sites[self.bonds])
        self.particles.set_offsets(sites)
        self.dipoles.set_offsets(sites)
        self.dipoles.set_UVC(*(coordinates(dipoles) * self.dipole_scale).T)
        self.field.set_UVC(*(coordinates(applied) * self.field_scale))
        return self.figure


def draw_lattice(rest: Vector, bonds: np.ndarray, positions: Vector, dipoles: Vector,
                 applied: Vector, dipole_scale: float, field_scale: float) -> plt.Figure:
    view = LatticeView(rest, bonds, dipole_scale, field_scale)
    return view.update(positions, dipoles, applied)


def animate_lattice(rest: Vector, bonds: np.ndarray, positions: Vector, dipoles: Vector,
                    applied: Vector, dipole_scale: float, field_scale: float) -> list[np.ndarray]:
    view = LatticeView(rest, bonds, dipole_scale, field_scale)
    frames = [capture(view.update(*state)) for state in zip(positions, dipoles, applied)]
    plt.close(view.figure)
    return frames


def draw_loading(strengths: np.ndarray, parallel: Scalar, transverse: Scalar) -> plt.Figure:
    figure, ax = plt.subplots(figsize=(6, 3.8))
    along = np.asarray(parallel.to_array()) * 100
    across = np.asarray(transverse.to_array()) * 100
    ax.plot(strengths, along, color=BOND, linewidth=2.3, label="along the field")
    ax.plot(strengths, across, color=TRANSVERSE, linewidth=2.3, label="across the field")
    ax.axhline(0, color=REST, linewidth=0.8)
    ax.set(xlabel="Applied field strength", ylabel="Change in rms extent (%)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    figure.tight_layout()
    return figure
