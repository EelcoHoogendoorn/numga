"""Drawing for the flake: the levels against the mass, a state's density on the flake, and the spin
running around the edge."""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from examples.util.animation import capture
from examples.quantum.kane_mele import core

# The bonds' shade, and the largest dot, in points squared, at the densest atom.
BONDS = "#d5d8dc"
DOT = 160.0


def coordinates(points: core.Point) -> np.ndarray:
    """The x and y coordinates of points in the plane."""
    return points.cast(core.ga.subspace("yz zx")).kernel


def draw_bonds(ax, flake: core.Flake) -> None:
    """The flake's bonds, faint, with the axes fitted to it."""
    positions = coordinates(flake.positions)
    ax.add_collection(LineCollection(positions[flake.first], colors=BONDS, linewidths=0.8, zorder=0))
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.axis("off")


def draw_levels(masses: np.ndarray, energies: core.Scalar, shares: core.Scalar, transition: float) -> plt.Figure:
    """The levels nearest zero against the mass, shaded by how much of each lies on the rim, with the
    mass where the bulk gap closes."""
    values, rims = energies.to_array(), shares.to_array()       # [masses, levels]
    figure, ax = plt.subplots(figsize=(6, 5), layout="constrained")
    dots = ax.scatter(np.repeat(masses, values.shape[1]), values.ravel(), c=rims.ravel(), cmap="plasma", vmin=0, vmax=1,
                      s=8, clip_on=False)
    ax.axvline(transition, color="0.4", linestyle="--", linewidth=1)
    # The energy's zero, which the levels on the rim cross until the gap reopens.
    ax.axhline(0, color="black", linewidth=0.8, zorder=0)
    ax.set(xlim=(masses[0], masses[-1]), xticks=[], yticks=[], xlabel="mass / hop", ylabel="energy / hop")
    ax.spines[["top", "right"]].set_visible(False)
    # Where each level lives, named at the colours' ends.
    shares = figure.colorbar(dots, ax=ax, ticks=[0, 1])
    shares.ax.set_yticklabels(["bulk", "rim"])
    return figure


def draw_states(flake: core.Flake, densities: core.Scalar) -> plt.Figure:
    """Each density on the flake, as dots sized and shaded by it."""
    positions, values = coordinates(flake.positions), densities.to_array()   # [atoms, 2], [cases, atoms]
    figure, axes = plt.subplots(1, len(values), figsize=(5 * len(values), 5))
    for ax, value in zip(np.atleast_1d(axes), values):
        draw_bonds(ax, flake)
        scaled = value / value.max()
        ax.scatter(*positions.T, s=DOT * scaled, c=scaled, cmap="magma_r", vmin=0, vmax=1)
    return figure


def spin_dots(spin: core.Vector) -> tuple[np.ndarray, np.ndarray]:
    """A spin density's dots: their areas, in proportion to the density, and their shades, the spin up
    over the largest density."""
    vectors = spin.cast(core.ga.subspace("x y z")).kernel
    density = np.linalg.norm(vectors, axis=-1)
    return DOT * density / density.max(), vectors[:, 2] / density.max()


def draw_spin(flake: core.Flake, spin: core.Vector) -> plt.Figure:
    """A spin density on the flake: dots sized by the density, red where the spin is up and blue
    where it is down."""
    sizes, shades = spin_dots(spin)
    figure, ax = plt.subplots(figsize=(6, 6))
    draw_bonds(ax, flake)
    ax.scatter(*coordinates(flake.positions).T, s=sizes, c=shades, cmap="RdBu_r", vmin=-1, vmax=1)
    return figure


def animate_spin(flake: core.Flake, spins: core.Vector) -> Iterator[np.ndarray]:
    """Each frame of the spin density, the flake's bonds drawn once and only the dots changing."""
    figure = draw_spin(flake, spins[0])
    dots = figure.axes[0].collections[-1]
    for spin in spins:
        sizes, shades = spin_dots(spin)
        dots.set_sizes(sizes)
        dots.set_array(shades)
        yield capture(figure)
    plt.close(figure)
