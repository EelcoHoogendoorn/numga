"""The reference lattice, its deformed bonds and dipoles, and its loading curve."""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.pyplot as plt
import numpy as np

from numga import stack

from examples.util import charts
from examples.util.animation import capture
from examples.util.wgsl import simplices
from examples.util.wgsl.camera import offset_panel
from examples.electromagnetism.electrostriction.core import Field, Point, Scalar, field_along

Topology = simplices.Topology

rest_colour = np.array([0xc4, 0xcb, 0xd0]) / 255
bond_colour = np.array([0x28, 0x65, 0x80]) / 255
dipole_colour = np.array([0xdf, 0x64, 0x35]) / 255
applied_colour = np.array([0xbd, 0x92, 0x24]) / 255
transverse_colour = "#8271a6"
label_colour = "0.4"
margin = 0.55
field_gap = 0.8
# Widths, dashes and radii in pixels, for a lattice picture about 300 pixels a side: an arrow's width and the
# length of the strokes of the chevron at its tip.
bond_width = 1.5
rest_width = 1.2
rest_dashes = (4.0, 3.0)
particle_radius = 2.2
rest_radius = 2.0
dipole_width, dipole_chevron = 2.6, 10.0
applied_width, applied_chevron = 3.5, 14.0


def lattice_axes(rest: Point) -> tuple[plt.Figure, plt.Axes, Scalar]:
    """A figure framing the reference sites, with room below them for the applied field; and half the box about the
    sites, across and up, the margin included."""
    mv = rest.context.multivector
    readings = stack((mv.x, mv.y))[:, None] & rest.batch()                              # [2, particles] Scalar+
    half_size = readings.abs().max(axis=-1) + margin                                     # [2] Scalar+
    limits = stack((-half_size, half_size), axis=-1).to_array()                          # [2, 2]
    figure, ax = plt.subplots(figsize=(3.4, 3.4), dpi=90)
    figure.subplots_adjust(0.02, 0.03, 0.98, 0.98)
    ax.set(xlim=limits[0], ylim=(limits[1, 0] - field_gap - 0.35, limits[1, 1]), aspect="equal")
    ax.axis("off")
    return figure, ax, half_size


def draw_rest(rest: Point, bonds: np.ndarray) -> plt.Figure:
    """The reference lattice: its particles and the bonds between them."""
    figure, ax, _ = lattice_axes(rest)
    mv = rest.context.multivector
    camera = offset_panel(ax, mv.x, mv.y)
    sites = rest.batch()
    simplices.overlay(ax, [simplices.strokes(sites, Topology(len(sites), bonds), bond_colour, bond_width, camera),
                           simplices.markers(sites, bond_colour, particle_radius, camera)])
    return figure


class LatticeView:
    """A lattice at actual scale over its reference, pinned once and redrawn per state; the arrows have fixed scales
    across frames."""

    def __init__(self, rest: Point, bonds: np.ndarray, dipole_scale: float, field_scale: float) -> None:
        self.dipole_scale, self.field_scale = dipole_scale, field_scale
        self.figure, ax, half_size = lattice_axes(rest)
        mv = rest.context.multivector
        self.camera = offset_panel(ax, mv.x, mv.y)
        sites = rest.batch()
        n_particles = len(sites)
        self.bonds = Topology(n_particles, bonds)
        # Each dipole a chain of two points, its tail and its head, about its particle.
        self.arrows = Topology.chains([2] * n_particles)
        # The applied field's arrow, centred below the lattice.
        self.anchor = mv.y.dual() * (-half_size[1] - field_gap / 2)                     # [] Point-
        self.reference = [simplices.strokes(sites, self.bonds, rest_colour, rest_width, self.camera,
                                            dashes=rest_dashes),
                          simplices.markers(sites, rest_colour, rest_radius, self.camera)]
        self.image = simplices.overlay(ax, self.reference)

    def update(self, positions: Point, dipoles: Point, applied: Field) -> plt.Figure:
        sites, arrows = positions.batch(), dipoles.batch() * self.dipole_scale
        # Each arrow a stroke with a chevron at its tip.
        dipole_ends = stack((sites - arrows / 2, sites + arrows / 2), axis=-1).reshape(-1)
        # The applied field drawn along the direction whose field it is.
        applied_arrow = field_along.solve(applied) * self.field_scale                   # [] Point-
        applied_ends = stack((self.anchor - applied_arrow / 2, self.anchor + applied_arrow / 2))
        self.image.set_data(simplices.composited(self.reference + [
            simplices.strokes(sites, self.bonds, bond_colour, bond_width, self.camera),
            simplices.markers(sites, bond_colour, particle_radius, self.camera),
            simplices.strokes(dipole_ends, self.arrows, dipole_colour, dipole_width, self.camera,
                              chevrons=np.ones(len(sites)), chevron=dipole_chevron),
            simplices.strokes(applied_ends, Topology.chains([2]), applied_colour, applied_width, self.camera,
                              chevrons=np.ones(1), chevron=applied_chevron)]))
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


@charts.style()
def draw_loading(strengths: np.ndarray, parallel: Scalar, transverse: Scalar) -> plt.Figure:
    # Half the column: two smooth curves, named axes and no numbers.
    figure, ax = charts.subplots(400, 260)
    along = np.asarray(parallel.to_array()) * 100
    across = np.asarray(transverse.to_array()) * 100
    ax.plot(strengths, along, color=bond_colour, label="along the field")
    ax.plot(strengths, across, color=transverse_colour, label="across the field")
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
