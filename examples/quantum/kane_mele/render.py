"""Figures and animation frames for the Kane–Mele graphene flake: sublattice geometry, energy
spectrum across staggered mass, eigenstate densities, and helical spin transport along the rim."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from numga import stack
from examples.util.animation import capture
from examples.quantum.kane_mele import core, scenarios
from examples.util import charts
from numga.render import drawing, simplices
from examples.util.panels import offset_panel, overlay

# Bond colour and stroke width in pixels, sublattice marker colours, and peak marker radius as a panel fraction.
bond_colour, bond_width = np.array(to_rgb("#d5d8dc")), 1.0
sublattice_colours = ("#4a6fa5", "#c8553d")
dot_radius = 0.0225
# Native pixel width, figure size in inches, and fractional margin around the flake.
flake_pixels = 300
flake_inches = 6.0
panel_margin = 0.05


def bond_layer(ax, flake: scenarios.Flake) -> np.ndarray:
    """Fits `ax` to `flake` and renders its nearest-neighbour bond strokes as an RGBA layer."""
    V, E = flake.bonds.n_elements
    positions = flake.positions.batch()                                        # [V] Point-
    mv = positions.context.multivector
    # Read horizontal and vertical extents by pairing atom points against the x and y coordinate planes.
    readings = stack((mv.x, mv.y))[:, None] & positions                        # [2, V] Scalar+
    low, high = readings.min(axis=-1), readings.max(axis=-1)                   # [2] Scalar+, [2] Scalar+
    spare = (high - low) * panel_margin                                        # [2] Scalar+
    limits = stack((low - spare, high + spare), axis=-1).to_array()            # [2, 2]
    ax.set(xlim=limits[0], ylim=limits[1], aspect="equal")
    ax.axis("off")
    camera = offset_panel(ax, core.mv.x, core.mv.y)
    return simplices.strokes(positions, flake.bonds, bond_colour, bond_width, camera)


def flake_figure(flake: scenarios.Flake) -> plt.Figure:
    """Draws the armchair honeycomb flake with bonds and sublattice-coloured atom markers."""
    V, E = flake.bonds.n_elements
    figure, ax = plt.subplots(figsize=(flake_inches, flake_inches), dpi=flake_pixels / flake_inches)
    bonds = bond_layer(ax, flake)
    palette = np.array([to_rgb(shade) for shade in sublattice_colours])
    shades = palette[(flake.sublattice.batch() < 0).astype(int)]               # [V, 3]
    positions = flake.positions.batch()                                        # [V] Point-
    camera = offset_panel(ax, core.mv.x, core.mv.y)
    atoms = simplices.markers(positions, shades, dot_radius * camera.pixels * np.sqrt(1 / 12), camera)
    overlay(ax, [bonds, atoms])
    return figure


@charts.style()
def levels_figure(masses: np.ndarray, energies: core.Scalar, shares: core.Scalar, transition: float) -> plt.Figure:
    """Plots the energy levels nearest zero against staggered mass, coloured by rim localization."""
    values, rims = energies.to_array(), shares.to_array()                      # [..., ...]
    figure, ax = charts.subplots(420, 340, layout="constrained")
    ax.scatter(np.repeat(masses, values.shape[1]), values.ravel(), c=rims.ravel(), cmap="plasma", vmin=0, vmax=1,
               clip_on=False)
    ax.axvline(transition, color="0.4", linestyle="--")
    ax.axhline(0, color="black", linewidth=charts.FRAME_WIDTH, zorder=0)
    ax.set(xlim=(masses[0], masses[-1]), xticks=[], yticks=[], xlabel="mass / hop", ylabel="energy / hop")
    ax.spines[["top", "right"]].set_visible(False)
    return figure


def states_figure(flake: scenarios.Flake, densities: core.Scalar) -> plt.Figure:
    """Draws side-by-side flake panels with atom markers sized and shaded by state density."""
    V, E = flake.bonds.n_elements
    positions, values = flake.positions.batch(), densities.to_array()          # [V] Point-, [..., V]
    n_panels = len(values)
    figure, axes = plt.subplots(1, n_panels, figsize=(flake_inches * n_panels, flake_inches),
                                dpi=flake_pixels / flake_inches, squeeze=False)
    for ax, value in zip(axes[0], values):
        bonds = bond_layer(ax, flake)
        scaled = value / value.max()
        camera = offset_panel(ax, core.mv.x, core.mv.y)
        dots = simplices.markers(positions, plt.get_cmap("magma_r")(scaled)[:, :3],
                                 dot_radius * camera.pixels * np.sqrt(scaled), camera)
        overlay(ax, [bonds, dots])
    return figure


def spin_marker_scales(spin: core.Spin) -> tuple[np.ndarray, np.ndarray]:
    """Relative marker areas and normalized z-spin components read by pairing `spin` with `xy`."""
    mv = spin.context.multivector
    batches = spin.batch()                                                     # [...] Spin-
    density = batches.norm()                                                   # [...] Scalar+
    largest = density.max()                                                    # [] Scalar+
    return (density / largest).to_array(), ((mv.xy & batches) / largest).to_array()


def spin_animation(flake: scenarios.Flake, pixels: int, spins: Iterable[core.Spin]) -> Iterator[np.ndarray]:
    """Yields RGB frames of `spins` over `flake`, sizing markers by density and colouring spin-up red and spin-down blue."""
    V, E = flake.bonds.n_elements
    figure, ax = plt.subplots(figsize=(flake_inches, flake_inches), dpi=pixels / flake_inches)
    bonds = bond_layer(ax, flake)
    positions = flake.positions.batch()                                        # [V] Point-
    camera = offset_panel(ax, core.mv.x, core.mv.y)
    shown = overlay(ax, [bonds])
    for spin in spins:
        areas, shades = spin_marker_scales(spin)
        dots = simplices.markers(positions, plt.get_cmap("RdBu_r")((shades + 1) / 2)[:, :3],
                                 dot_radius * camera.pixels * np.sqrt(areas), camera)
        shown.set_data(drawing.composited([bonds, dots]))
        yield capture(figure)
    plt.close(figure)
