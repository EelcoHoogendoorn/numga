"""The reference lattice, its deformed bonds and dipoles, and its loading curve."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from numga import stack

from examples.animation import capture
from examples.electromagnetism.electrostriction.core import Field, Point, Scalar, mv, time

rest_colour = "#c4cbd0"
bond_colour = "#286580"
dipole_colour = "#df6435"
applied_colour = "#bd9224"
transverse_colour = "#8271a6"
label_colour = "0.4"
margin = 0.55
field_gap = 0.8


def coordinates(value: Field) -> np.ndarray:
    """Components of fields along x and y: their pairings with the planes those directions sweep in time."""
    sweeps = stack((mv.x.dual(), mv.y.dual())) & time                                   # [2] Antibivector
    return np.asarray((value[..., None] & sweeps).to_array())                            # [..., 2]


def point_coordinates(value: Point) -> np.ndarray:
    """Coordinates of positions and dipoles at every particle: their pairings with the lines x and y."""
    return np.asarray((stack((mv.x, mv.y)) & value.batch()[..., None]).to_array())   # [..., particles, 2]


def lattice_axes(sites: np.ndarray) -> tuple[plt.Figure, plt.Axes, np.ndarray]:
    """A figure framing the reference sites, with room below them for the applied field."""
    figure, ax = plt.subplots(figsize=(6, 6), dpi=90)
    reach = np.abs(sites).max(axis=0) + margin
    ax.set(xlim=(-reach[0], reach[0]), ylim=(-reach[1] - field_gap - 0.35, reach[1]), aspect="equal")
    ax.axis("off")
    figure.subplots_adjust(0.02, 0.03, 0.98, 0.98)
    return figure, ax, reach


def draw_rest(rest: Point, bonds: np.ndarray) -> plt.Figure:
    """The reference lattice: its particles and the bonds between them."""
    sites = point_coordinates(rest)
    figure, ax, _ = lattice_axes(sites)
    ax.add_collection(LineCollection(sites[bonds], colors=bond_colour, linewidths=2, zorder=2))
    ax.scatter(*sites.T, s=34, color=bond_colour, zorder=4)
    return figure


class LatticeView:
    """Artists for a lattice at actual scale; the arrows have fixed scales across frames."""

    def __init__(self, rest: Point, bonds: np.ndarray, dipole_scale: float, field_scale: float) -> None:
        self.bonds, self.dipole_scale, self.field_scale = bonds, dipole_scale, field_scale
        sites = point_coordinates(rest)
        self.figure, ax, reach = lattice_axes(sites)
        ax.add_collection(LineCollection(sites[bonds], colors=rest_colour, linewidths=1.2,
                                        linestyles="dashed", zorder=0))
        ax.scatter(*sites.T, s=22, facecolors="white", edgecolors=rest_colour, linewidths=1, zorder=1)
        self.links = LineCollection(sites[bonds], colors=bond_colour, linewidths=2, zorder=2)
        ax.add_collection(self.links)
        self.particles = ax.scatter(*sites.T, s=34, color=bond_colour, zorder=4)
        self.dipoles = ax.quiver(*sites.T, np.zeros(len(sites)), np.zeros(len(sites)),
                                 color=dipole_colour, angles="xy", scale_units="xy", scale=1,
                                 pivot="middle", width=0.009, headwidth=3, headlength=4, zorder=5)
        field_y = -reach[1] - field_gap / 2
        self.field = ax.quiver([0], [field_y], [0], [0], color=applied_colour, angles="xy",
                               scale_units="xy", scale=1, pivot="middle", width=0.012,
                               headwidth=3, headlength=4)

    def update(self, positions: Point, dipoles: Point, applied: Field) -> plt.Figure:
        sites = point_coordinates(positions)
        self.links.set_segments(sites[self.bonds])
        self.particles.set_offsets(sites)
        self.dipoles.set_offsets(sites)
        self.dipoles.set_UVC(*(point_coordinates(dipoles) * self.dipole_scale).T)
        self.field.set_UVC(*(coordinates(applied) * self.field_scale))
        return self.figure


def draw_lattice(rest: Point, bonds: np.ndarray, positions: Point, dipoles: Point,
                 applied: Field, dipole_scale: float, field_scale: float) -> plt.Figure:
    view = LatticeView(rest, bonds, dipole_scale, field_scale)
    return view.update(positions, dipoles, applied)


def animate_lattice(rest: Point, bonds: np.ndarray, positions: Point, dipoles: Point,
                    applied: Field, dipole_scale: float, field_scale: float) -> list[np.ndarray]:
    view = LatticeView(rest, bonds, dipole_scale, field_scale)
    frames = [capture(view.update(*state)) for state in zip(positions, dipoles, applied)]
    plt.close(view.figure)
    return frames


def draw_loading(strengths: np.ndarray, parallel: Scalar, transverse: Scalar) -> plt.Figure:
    figure, ax = plt.subplots(figsize=(6, 3.8))
    along = np.asarray(parallel.to_array()) * 100
    across = np.asarray(transverse.to_array()) * 100
    ax.plot(strengths, along, color=bond_colour, linewidth=2.3, label="along the field")
    ax.plot(strengths, across, color=transverse_colour, linewidth=2.3, label="across the field")
    # Axes through no field and no change, without tick marks or numbers, each named along itself: the field's above
    # its line, where the curves are not.
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_position(("data", 0))
    ax.set(xticks=[], yticks=[])
    ax.set_ylabel("change in rms extent", color=label_colour)
    ax.text(0.5, 0, "applied field strength", transform=ax.get_yaxis_transform(), ha="center", va="bottom",
            color=label_colour)
    ax.legend(loc="lower center", frameon=False)
    figure.tight_layout()
    return figure
