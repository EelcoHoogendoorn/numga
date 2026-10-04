"""Drawing for the flake: the levels against the mass, a state's density on the flake, and the spin
running around the edge."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from examples.animation import capture
from examples.quantum.kane_mele import core

# The bonds' shade, and the largest dot, in points squared, at the densest atom.
BONDS = "#d5d8dc"
DOT = 160.0


def coordinates(points: core.Vector) -> np.ndarray:
    """The x and y coordinates of points in the plane."""
    return points.cast(core.ga.subspace("x y")).kernel


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
    figure, ax = plt.subplots(figsize=(7, 5))
    dots = ax.scatter(np.repeat(masses, values.shape[1]), values.ravel(), c=rims.ravel(), cmap="plasma",
                      vmin=0, vmax=1, s=8)
    ax.axvline(transition, color="0.4", linestyle="--", linewidth=1)
    ax.set_xlabel("mass / hop")
    ax.set_ylabel("energy / hop")
    ax.set_title("levels nearest zero, spin up")
    figure.colorbar(dots, ax=ax, label="share on the rim")
    return figure


def draw_states(flake: core.Flake, densities: core.Scalar, titles: list[str]) -> plt.Figure:
    """Each density on the flake, as dots sized and shaded by it."""
    positions, values = coordinates(flake.positions), densities.to_array()   # [atoms, 2], [cases, atoms]
    figure, axes = plt.subplots(1, len(values), figsize=(5 * len(values), 5))
    for ax, value, title in zip(np.atleast_1d(axes), values, titles):
        draw_bonds(ax, flake)
        scaled = value / value.max()
        ax.scatter(*positions.T, s=DOT * scaled, c=scaled, cmap="magma_r", vmin=0, vmax=1)
        ax.set_title(title)
    return figure


def draw_spin(flake: core.Flake, spin: core.Vector, title: str) -> plt.Figure:
    """A spin density on the flake: dots sized by the density, red where the spin is up and blue
    where it is down."""
    positions, vectors = coordinates(flake.positions), spin.cast(core.ga.subspace("x y z")).kernel
    density = np.linalg.norm(vectors, axis=-1)
    figure, ax = plt.subplots(figsize=(6, 6))
    draw_bonds(ax, flake)
    ax.scatter(*positions.T, s=DOT * density / density.max(), c=vectors[:, 2] / density.max(), cmap="RdBu_r",
               vmin=-1, vmax=1)
    ax.set_title(title)
    return figure


def animate_spin(flake: core.Flake, spins: core.Vector) -> list[np.ndarray]:
    """Each frame of the spin density."""
    images = []
    for spin in spins:
        figure = draw_spin(flake, spin, "spin up red, spin down blue")
        images.append(capture(figure))
        plt.close(figure)
    return images
