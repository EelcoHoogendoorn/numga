"""Lattice panels and strain charts depict the reference patch, its induced dipoles, and its electrostrictive deformation."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np

from numga import stack
from numga.render import Camera, drawing, simplices
from numga.topology import Topology

from examples.electromagnetism.electrostriction.core import Field, Point, Scalar, field_along
from examples.util import charts
from examples.util.animation import capture
from examples.util.panels import offset_panel, overlay

rest_colour = np.array([0xc4, 0xcb, 0xd0]) / 255
bond_colour = np.array([0x28, 0x65, 0x80]) / 255
dipole_colour = np.array([0xdf, 0x64, 0x35]) / 255
applied_colour = np.array([0xbd, 0x92, 0x24]) / 255
transverse_colour = "#8271a6"
label_colour = "0.4"
margin = 0.55
field_gap = 0.8
# Stroke widths, dash lengths, marker radii, and arrowhead chevron lengths in pixels for a ~300 px panel.
bond_width = 1.5
rest_width = 1.2
rest_dashes = (4.0, 3.0)
particle_radius = 2.2
rest_radius = 2.0
dipole_width, dipole_chevron = 2.6, 10.0
applied_width, applied_chevron = 3.5, 14.0


def lattice_axes(rest_positions: Point) -> tuple[plt.Figure, plt.Axes, Scalar]:
    """Bounding coordinates of the reference sites frame the lattice panel with room below for the applied-field arrow (`[2] Scalar+`)."""
    mv = rest_positions.context.multivector
    readings = stack((mv.x, mv.y))[:, None] & rest_positions.batch()                    # [2, ...] Scalar+
    half_size = readings.abs().max(axis=-1) + margin                                    # [2] Scalar+
    limits = stack((-half_size, half_size), axis=-1).to_array()                         # [2, 2]
    figure, ax = plt.subplots(figsize=(3.4, 3.4), dpi=90)
    figure.subplots_adjust(0.02, 0.03, 0.98, 0.98)
    ax.set(xlim=limits[0], ylim=(limits[1, 0] - field_gap - 0.35, limits[1, 1]), aspect="equal")
    ax.axis("off")
    return figure, ax, half_size


def rest_figure(rest_positions: Point, topology: Topology) -> plt.Figure:
    """Markers and bond strokes show the undeformed hexagonal reference lattice."""
    figure, ax, _ = lattice_axes(rest_positions)
    mv = rest_positions.context.multivector
    camera = offset_panel(ax, mv.x, mv.y)
    sites = rest_positions.batch()                                                      # [...] Point-
    overlay(
        ax,
        [
            simplices.strokes(sites, topology, bond_colour, bond_width, camera),
            simplices.markers(sites, bond_colour, particle_radius, camera),
        ],
    )
    return figure


def state_layers(
    positions: Point,
    dipoles: Point,
    applied_field: Field,
    topology: Topology,
    arrow_topology: Topology,
    anchor: Point,
    camera: Camera,
    dipole_scale: float,
    field_scale: float,
) -> list[np.ndarray]:
    """Bond strokes, particle markers, dipole chevrons, and the applied-field arrow compose one equilibrium layer list."""
    V, E = topology.n_elements
    sites = positions.batch()                                                           # [V] Point-
    dipole_arrows = dipoles.batch() * dipole_scale                                      # [V] Point-
    dipole_ends = stack((sites - dipole_arrows / 2, sites + dipole_arrows / 2), axis=-1).reshape(-1)
    # Draw the applied field along the spatial direction whose spacetime dual produces it.
    field_arrow = field_along.solve(applied_field) * field_scale                        # [] Point-
    field_ends = stack((anchor - field_arrow / 2, anchor + field_arrow / 2))            # [2] Point-
    return [
        simplices.strokes(sites, topology, bond_colour, bond_width, camera),
        simplices.markers(sites, bond_colour, particle_radius, camera),
        simplices.strokes(
            dipole_ends, arrow_topology, dipole_colour, dipole_width, camera,
            chevrons=np.ones(V), chevron=dipole_chevron,
        ),
        simplices.strokes(
            field_ends, Topology.chains([2]), applied_colour, applied_width, camera,
            chevrons=np.ones(1), chevron=applied_chevron,
        ),
    ]


def lattice_figure(
    rest_positions: Point,
    topology: Topology,
    positions: Point,
    dipoles: Point,
    applied_field: Field,
    dipole_scale: float,
    field_scale: float,
) -> plt.Figure:
    """True-scale deformed bonds and dipole arrows overlay the dashed grey reference lattice."""
    V, E = topology.n_elements
    figure, ax, half_size = lattice_axes(rest_positions)
    mv = rest_positions.context.multivector
    camera = offset_panel(ax, mv.x, mv.y)
    rest_sites = rest_positions.batch()                                                 # [V] Point-
    anchor = mv.y.dual() * (-half_size[1] - field_gap / 2)                              # [] Point-
    arrow_topology = Topology.chains([2] * V)
    reference_layers = [
        simplices.strokes(rest_sites, topology, rest_colour, rest_width, camera, dashes=rest_dashes),
        simplices.markers(rest_sites, rest_colour, rest_radius, camera),
    ]
    overlay(
        ax,
        reference_layers + state_layers(
            positions, dipoles, applied_field, topology, arrow_topology,
            anchor, camera, dipole_scale, field_scale,
        ),
    )
    return figure


def lattice_animation(
    rest_positions: Point,
    topology: Topology,
    dipole_scale: float,
    field_scale: float,
    states: Iterable[tuple[Point, Point, Field]],
) -> Iterator[np.ndarray]:
    """Yields RGB frames of the deformed lattice and induced dipoles as the applied field turns."""
    V, E = topology.n_elements
    figure, ax, half_size = lattice_axes(rest_positions)
    mv = rest_positions.context.multivector
    camera = offset_panel(ax, mv.x, mv.y)
    rest_sites = rest_positions.batch()                                                 # [V] Point-
    anchor = mv.y.dual() * (-half_size[1] - field_gap / 2)                              # [] Point-
    arrow_topology = Topology.chains([2] * V)
    reference_layers = [
        simplices.strokes(rest_sites, topology, rest_colour, rest_width, camera, dashes=rest_dashes),
        simplices.markers(rest_sites, rest_colour, rest_radius, camera),
    ]
    image = overlay(ax, reference_layers)
    for positions, dipoles, applied_field in states:
        layers = state_layers(
            positions, dipoles, applied_field, topology, arrow_topology,
            anchor, camera, dipole_scale, field_scale,
        )
        image.set_data(drawing.composited(reference_layers + layers))
        yield capture(figure)
    plt.close(figure)


@charts.style()
def loading_figure(strengths: np.ndarray, parallel_strain: Scalar, transverse_strain: Scalar) -> plt.Figure:
    """Longitudinal and transverse percentage strain curves track electrostriction across increasing field strength."""
    figure, ax = charts.subplots(400, 260)
    parallel_percent = np.asarray(parallel_strain.to_array()) * 100
    transverse_percent = np.asarray(transverse_strain.to_array()) * 100
    ax.plot(strengths, parallel_percent, color=bond_colour)
    ax.plot(strengths, transverse_percent, color=transverse_colour)
    # Zero-crossed axes without tick numbers, with the horizontal axis named above zero clear of the contraction curve.
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_position(("data", 0))
    ax.set(xticks=[], yticks=[])
    ax.set_ylabel("change in rms extent", color=label_colour)
    ax.text(
        0.5, 0, "applied field strength",
        transform=ax.get_yaxis_transform(), ha="center", va="bottom", color=label_colour,
    )
    figure.tight_layout()
    return figure
