"""Drawing for the flake: the levels against the mass, a state's density on the flake, and the spin
running around the edge."""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from numga import stack
from examples.util.animation import capture
from examples.quantum.kane_mele import core
from examples.util import charts
from examples.util.wgsl import simplices
from examples.util.wgsl.camera import flat_panel

# The bonds' shade and width in pixels, the sublattices' shades, and the largest dot's radius, at the densest atom, as
# a share of the panel's width.
BONDS, BOND_WIDTH = np.array(to_rgb("#d5d8dc")), 1.0
SUBLATTICES = ("#4a6fa5", "#c8553d")
DOT = 0.0225
# The pixels across a picture of the flake, and the inches the flake is drawn at; the room past the flake on either
# side, as a share of its reach.
PIXELS = 300
INCHES = 6.0
MARGIN = 0.05


def draw_bonds(ax, flake: core.Flake) -> np.ndarray:
    """The axes fitted to the flake, and its bonds, faint: their picture, for the atoms to be laid over."""
    positions = flake.positions.batch()                                       # [atoms] Point-
    mv = positions.context.multivector
    # The flake's reach across and up, read by pairing with the planes x and y, and a margin past it either side.
    readings = stack((mv.x, mv.y))[:, None] & positions                       # [2, atoms] Scalar+
    low, high = readings.min(axis=-1), readings.max(axis=-1)                  # [2] Scalar+, Scalar+
    spare = (high - low) * MARGIN
    limits = stack((low - spare, high + spare), axis=-1).to_array()
    ax.set(xlim=limits[0], ylim=limits[1], aspect="equal")
    ax.axis("off")
    camera = flat_panel(ax, positions.algebra)
    return simplices.strokes(positions, simplices.Mesh(len(positions), flake.first), BONDS, BOND_WIDTH, camera)


def draw_flake(flake: core.Flake) -> plt.Figure:
    """The flake's bonds, and its atoms shaded by sublattice."""
    figure, ax = plt.subplots(figsize=(INCHES, INCHES), dpi=PIXELS / INCHES)
    bonds = draw_bonds(ax, flake)
    shades = np.array([to_rgb(shade) for shade in SUBLATTICES])[(flake.sublattice < 0).astype(int)]
    positions = flake.positions.batch()                                       # [atoms] Point-
    # The atoms' dots, the area of a twelfth of the densest's.
    camera = flat_panel(ax, positions.algebra)
    atoms = simplices.markers(positions, shades, DOT * camera.pixels * np.sqrt(1 / 12), camera)
    simplices.overlay(ax, [bonds, atoms])
    return figure


@charts.style()
def draw_levels(masses: np.ndarray, energies: core.Scalar, shares: core.Scalar, transition: float) -> plt.Figure:
    """The levels nearest zero against the mass, shaded by how much of each lies on the rim, with the
    mass where the bulk gap closes."""
    values, rims = energies.to_array(), shares.to_array()       # [masses, levels]
    # Half the column: the levels as rows of dots over the masses, tall enough for them to stand apart.
    figure, ax = charts.subplots(420, 340, layout="constrained")
    ax.scatter(np.repeat(masses, values.shape[1]), values.ravel(), c=rims.ravel(), cmap="plasma", vmin=0, vmax=1,
               clip_on=False)
    ax.axvline(transition, color="0.4", linestyle="--")
    # The energy's zero, which the levels on the rim cross until the gap reopens.
    ax.axhline(0, color="black", linewidth=charts.FRAME_WIDTH, zorder=0)
    ax.set(xlim=(masses[0], masses[-1]), xticks=[], yticks=[], xlabel="mass / hop", ylabel="energy / hop")
    ax.spines[["top", "right"]].set_visible(False)
    return figure


def draw_states(flake: core.Flake, densities: core.Scalar) -> plt.Figure:
    """Each density on the flake, as dots sized and shaded by it."""
    positions, values = flake.positions.batch(), densities.to_array()         # [atoms] Point-, [cases, atoms]
    figure, axes = plt.subplots(1, len(values), figsize=(INCHES * len(values), INCHES), dpi=PIXELS / INCHES,
                                squeeze=False)
    for ax, value in zip(axes[0], values):
        bonds = draw_bonds(ax, flake)
        scaled = value / value.max()
        camera = flat_panel(ax, positions.algebra)
        # Each dot's area in proportion to its density.
        dots = simplices.markers(positions, plt.get_cmap("magma_r")(scaled)[:, :3],
                                 DOT * camera.pixels * np.sqrt(scaled), camera)
        simplices.overlay(ax, [bonds, dots])
    return figure


def spin_dots(spin: core.Spin) -> tuple[np.ndarray, np.ndarray]:
    """A spin density's dots: their areas over the largest's, in proportion to the density, and their shades, the spin
    up over the largest density."""
    mv = spin.context.multivector
    density = spin.batch().norm()                                            # [atoms] Scalar+
    largest = density.max()
    return (density / largest).to_array(), ((mv.xy & spin.batch()) / largest).to_array()


def animate_spin(flake: core.Flake, spins: core.Spin, pixels: int) -> Iterator[np.ndarray]:
    """Each frame of the spin density on the flake, so many pixels across: dots sized by the density, red where the
    spin is up and blue where it is down, over the flake's bonds, drawn once."""
    figure, ax = plt.subplots(figsize=(INCHES, INCHES), dpi=pixels / INCHES)
    bonds = draw_bonds(ax, flake)
    positions = flake.positions.batch()                                       # [atoms] Point-
    camera = flat_panel(ax, positions.algebra)
    shown = simplices.overlay(ax, [bonds])
    for spin in spins:
        areas, shades = spin_dots(spin)
        dots = simplices.markers(positions, plt.get_cmap("RdBu_r")((shades + 1) / 2)[:, :3],
                                 DOT * camera.pixels * np.sqrt(areas), camera)
        shown.set_data(simplices.composited([bonds, dots]))
        yield capture(figure)
    plt.close(figure)
