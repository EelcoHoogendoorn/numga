"""The reference lattice, its deformed bonds and dipoles, and its loading curve."""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from numga import stack

from examples.util.animation import capture
from examples.electromagnetism.electrostriction.core import Field, Point, Scalar

rest_colour = "#c4cbd0"
bond_colour = "#286580"
dipole_colour = "#df6435"
applied_colour = "#bd9224"
transverse_colour = "#8271a6"
label_colour = "0.4"
margin = 0.55
field_gap = 0.8
# Line widths and marker sizes, in points, for a lattice picture about 300 pixels a side.
bond_width = 1.1
rest_width = 0.7
particle_size = 11
rest_size = 7


def coordinates(value: Field) -> np.ndarray:
    """Components of fields along x and y: their pairings with the planes those directions sweep in time."""
    mv = value.context.multivector
    time = -mv.t.dual()                                                                  # [] Antivector-
    sweeps = stack((mv.x.dual(), mv.y.dual())) & time                                   # [2] Antibivector-
    return np.asarray((value[..., None] & sweeps).to_array())                            # [..., 2]


def point_coordinates(value: Point) -> np.ndarray:
    """Coordinates of positions and dipoles at every particle: their pairings with the lines x and y."""
    mv = value.context.multivector
    return np.asarray((stack((mv.x, mv.y)) & value.batch()[..., None]).to_array())   # [..., particles, 2]


def lattice_axes(sites: np.ndarray) -> tuple[plt.Figure, plt.Axes, np.ndarray]:
    """A figure framing the reference sites, with room below them for the applied field."""
    figure, ax = plt.subplots(figsize=(3.4, 3.4), dpi=90)
    half_size = np.abs(sites).max(axis=0) + margin
    ax.set(xlim=(-half_size[0], half_size[0]), ylim=(-half_size[1] - field_gap - 0.35, half_size[1]),
           aspect="equal")
    ax.axis("off")
    figure.subplots_adjust(0.02, 0.03, 0.98, 0.98)
    return figure, ax, half_size


def draw_rest(rest: Point, bonds: np.ndarray) -> plt.Figure:
    """The reference lattice: its particles and the bonds between them."""
    sites = point_coordinates(rest)
    figure, ax, _ = lattice_axes(sites)
    ax.add_collection(LineCollection(sites[bonds], colors=bond_colour, linewidths=bond_width, zorder=2))
    ax.scatter(*sites.T, s=particle_size, color=bond_colour, zorder=4)
    return figure


class LatticeView:
    """Artists for a lattice at actual scale; the arrows have fixed scales across frames."""

    def __init__(self, rest: Point, bonds: np.ndarray, dipole_scale: float, field_scale: float) -> None:
        self.bonds, self.dipole_scale, self.field_scale = bonds, dipole_scale, field_scale
        sites = point_coordinates(rest)
        self.figure, ax, half_size = lattice_axes(sites)
        ax.add_collection(LineCollection(sites[bonds], colors=rest_colour, linewidths=rest_width,
                                        linestyles="dashed", zorder=0))
        ax.scatter(*sites.T, s=rest_size, facecolors="white", edgecolors=rest_colour, linewidths=rest_width,
                   zorder=1)
        self.links = LineCollection(sites[bonds], colors=bond_colour, linewidths=bond_width, zorder=2)
        ax.add_collection(self.links)
        self.particles = ax.scatter(*sites.T, s=particle_size, color=bond_colour, zorder=4)
        self.dipoles = ax.quiver(*sites.T, np.zeros(len(sites)), np.zeros(len(sites)),
                                 color=dipole_colour, angles="xy", scale_units="xy", scale=1,
                                 pivot="middle", width=0.009, headwidth=3, headlength=4, zorder=5)
        field_y = -half_size[1] - field_gap / 2
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
                    applied: Field, dipole_scale: float, field_scale: float) -> Iterator[np.ndarray]:
    view = LatticeView(rest, bonds, dipole_scale, field_scale)
    for state in zip(positions, dipoles, applied):
        yield capture(view.update(*state))
    plt.close(view.figure)


def draw_loading(strengths: np.ndarray, parallel: Scalar, transverse: Scalar) -> plt.Figure:
    figure, ax = plt.subplots(figsize=(3.4, 2.15), dpi=90)
    along = np.asarray(parallel.to_array()) * 100
    across = np.asarray(transverse.to_array()) * 100
    ax.plot(strengths, along, color=bond_colour, linewidth=1.3, label="along the field")
    ax.plot(strengths, across, color=transverse_colour, linewidth=1.3, label="across the field")
    # Axes through no field and no change, without tick marks or numbers, each named along itself: the field's above
    # its line, where the curves are not.
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_position(("data", 0))
    ax.spines[["left", "bottom"]].set_linewidth(0.5)
    ax.set(xticks=[], yticks=[])
    ax.set_ylabel("change in rms extent", color=label_colour, fontsize=7)
    ax.text(0.5, 0, "applied field strength", transform=ax.get_yaxis_transform(), ha="center", va="bottom",
            color=label_colour, fontsize=7)
    ax.legend(loc="lower center", frameon=False, fontsize=7)
    figure.tight_layout()
    return figure
