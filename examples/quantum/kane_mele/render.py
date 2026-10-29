"""Drawing for the flake: the levels against the mass, a state's density on the flake, and the spin
running around the edge."""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from numga import stack
from examples.util.animation import capture
from examples.quantum.kane_mele import core
from examples.util import charts

# The bonds' shade, the sublattices' shades, and the largest dot, in points squared on a flake drawn six
# inches across, at the densest atom.
BONDS = "#d5d8dc"
SUBLATTICES = ("#4a6fa5", "#c8553d")
DOT = 160.0
# The pixels across a picture of the flake, and the inches the flake is drawn at.
PIXELS = 300
INCHES = 6.0


def coordinates(points: core.Point) -> np.ndarray:
    """The x and y coordinates of points in the plane."""
    mv = points.context.multivector
    return (stack((mv.x, mv.y)) & points.batch()[..., None]).to_array()


def draw_bonds(ax, flake: core.Flake) -> None:
    """The flake's bonds, faint, with the axes fitted to it."""
    positions = coordinates(flake.positions)
    ax.add_collection(LineCollection(positions[flake.first], colors=BONDS, linewidths=0.8, zorder=0))
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.axis("off")


def draw_flake(flake: core.Flake) -> plt.Figure:
    """The flake's bonds, and its atoms shaded by sublattice."""
    figure, ax = plt.subplots(figsize=(INCHES, INCHES), dpi=PIXELS / INCHES)
    draw_bonds(ax, flake)
    shades = np.array(SUBLATTICES)[(flake.sublattice < 0).astype(int)]
    ax.scatter(*coordinates(flake.positions).T, s=DOT / 12, c=shades)
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
    positions, values = coordinates(flake.positions), densities.to_array()   # [atoms, 2], [cases, atoms]
    figure, axes = plt.subplots(1, len(values), figsize=(INCHES * len(values), INCHES), dpi=PIXELS / INCHES,
                                squeeze=False)
    for ax, value in zip(axes[0], values):
        draw_bonds(ax, flake)
        scaled = value / value.max()
        ax.scatter(*positions.T, s=DOT * scaled, c=scaled, cmap="magma_r", vmin=0, vmax=1)
    return figure


def spin_dots(spin: core.Spin) -> tuple[np.ndarray, np.ndarray]:
    """A spin density's dots: their areas, in proportion to the density, and their shades, the spin up
    over the largest density."""
    mv = spin.context.multivector
    # Each spin read by the planes it turns in, yz, zx and xy.
    components = (stack((mv.yz, mv.zx, mv.xy)) & spin.batch()[..., None]).to_array()   # [atoms, 3]
    density = np.linalg.norm(components, axis=-1)
    return DOT * density / density.max(), components[:, 2] / density.max()


def draw_spin(flake: core.Flake, spin: core.Spin, pixels: int) -> plt.Figure:
    """A spin density on the flake, so many pixels across: dots sized by the density, red where the spin
    is up and blue where it is down."""
    sizes, shades = spin_dots(spin)
    figure, ax = plt.subplots(figsize=(INCHES, INCHES), dpi=pixels / INCHES)
    draw_bonds(ax, flake)
    ax.scatter(*coordinates(flake.positions).T, s=sizes, c=shades, cmap="RdBu_r", vmin=-1, vmax=1)
    return figure


def animate_spin(flake: core.Flake, spins: core.Spin, pixels: int) -> Iterator[np.ndarray]:
    """Each frame of the spin density, the flake's bonds drawn once and only the dots changing, the figure so many
    pixels across."""
    figure = draw_spin(flake, spins[0], pixels)
    dots = figure.axes[0].collections[-1]
    for spin in spins:
        sizes, shades = spin_dots(spin)
        dots.set_sizes(sizes)
        dots.set_array(shades)
        yield capture(figure)
    plt.close(figure)
