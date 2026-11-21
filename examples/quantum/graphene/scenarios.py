"""Scenes for graphene: the two bands over the Brillouin zone, the pseudospin around the two
valleys, and the Berry phase of loops around a valley, with and without a gap.

Momenta are in inverse carbon-carbon distances, 1 / 0.142 nm, and energies in eV.
"""

from __future__ import annotations

from functools import partial

import numpy as np

from numga import dim, stack
from examples.quantum.graphene import core

mv = core.mv
# The slope of the cones, in eV per inverse carbon-carbon distance (`energy * length`).
cone_slope = 1.5 * core.hopping * core.bond_length
# The bands: the half width of the square of momenta that holds the Brillouin zone, its samples each way,
# and no gap.
zone_extent = mv.scalar(4.5, unit=1 / dim.length)
n_zone_samples = 121
gapless = mv.scalar(0.0, unit=dim.energy)
# The pseudospin about the valleys: the half width of the square about each, its samples each way, and the gap.
valley_extent = mv.scalar(0.5, unit=1 / dim.length)
n_valley_samples = 11
texture_gap = mv.scalar(0.3, unit=dim.energy)
# The Berry phase: the loops' radii in inverse carbon-carbon distances, the gaps in eV, and the steps around.
n_radii = 30
radii = mv.scalar(np.linspace(0.01, 0.6, n_radii)[:, None], unit=1 / dim.length)
n_gaps = 4
gaps = mv.scalar(np.array([0.0, 0.2, 0.5, 1.0])[:, None], unit=dim.energy)
n_loop_steps = 400


# --- math -----------------------------------------------------------------------------
def zone_bands(extent: core.Scalar, n_samples: int, gap: core.Scalar) -> tuple[core.Wavevector, core.Field, core.Scalar]:
    """The pseudospin field and Hamiltonian eigenvalues fill a `[n_samples, n_samples]` momentum grid across the Brillouin zone."""
    grid_axis = np.linspace(-1.0, 1.0, n_samples) * extent
    kx, ky = grid_axis[None, :], grid_axis[:, None]
    momenta = mv.x * kx + mv.y * ky                                             # [n_samples, n_samples] Wavevector+
    field = core.pseudospin_field(momenta, gap)                                 # [n_samples, n_samples] Field+
    energies, _ = core.hamiltonian(field).eigh()                                # [n_samples, n_samples, 4] Scalar+
    return momenta, field, energies


def valley_textures(extent: core.Scalar, n_samples: int, gap: core.Scalar) -> tuple[core.Wavevector, core.Pseudospin]:
    """Upper-band pseudospins (`[2, n_samples, n_samples] Pseudospin-`) sample a momentum square around each valley."""
    grid_axis = np.linspace(-1.0, 1.0, n_samples) * extent
    kx, ky = grid_axis[None, :], grid_axis[:, None]
    offsets = mv.x * kx + mv.y * ky                                             # [n_samples, n_samples] Wavevector+
    return offsets, core.upper_pseudospins(offsets, gap)


def loop_berry(loop_radii: core.Scalar, sublattice_gaps: core.Scalar, n_steps: int) -> tuple[core.Rotor, core.Pseudospin]:
    """Transports a pseudospin frame around momentum circles at both valleys, giving the loop holonomies and starting pseudospins."""
    circle = core.momentum_circle(n_steps)                                      # [n_steps + 1] Wavevector+
    loops = core.valleys[:, None] + circle[:, None, None] * loop_radii          # [n_steps + 1, 2, ...] Wavevector+
    field = core.pseudospin_field(loops[:, None], sublattice_gaps[:, None, None])   # [n_steps + 1, ..., 2, ...] Field+
    # The upper band's pseudospin, along the field.
    directions = field.normalized().dual()                                      # [n_steps + 1, ..., 2, ...] Pseudospin-
    return core.transport(directions)[-1], directions[0]


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_figure
    from examples.quantum.graphene import render

    momenta, field, energies = zone_bands(zone_extent, n_zone_samples, gapless)
    offsets, valley_directions = valley_textures(valley_extent, n_valley_samples, texture_gap)
    holonomies, start_directions = loop_berry(radii, gaps, n_loop_steps)
    save_figure(render.field_figure(momenta, field), "graphene_field")
    save_figure(render.bands_figure(momenta, energies), "graphene_bands")
    save_figure(render.textures_figure(partial(core.upper_pseudospins, gap=texture_gap), valley_extent), "graphene_pseudospin")
    save_figure(render.berry_figure(radii, gaps, holonomies, start_directions), "graphene_berry")

    # checks
    # The energies are minus and plus the field's length, each twice; without a gap the field
    # vanishes at the valleys, where the cones meet.
    length = field.norm()                                                       # [n_zone_samples, n_zone_samples] Scalar+
    expected_energies = stack((-length, -length, length, length), axis=-1)      # [n_zone_samples, n_zone_samples, 4] Scalar+
    assert ((energies - expected_energies).abs() < mv.scalar(1e-9, unit=dim.energy)).all()
    assert (core.pseudospin_field(core.valleys, gapless).norm() < mv.scalar(1e-6, unit=dim.energy)).all()
    # The upper band's pseudospin points along the field.
    valley_field = core.pseudospin_field(core.valleys[:, None, None] + offsets, texture_gap)   # [2, n_valley_samples, n_valley_samples] Field+
    assert ((valley_directions - valley_field.normalized().dual()).antinorm() < 1e-9).all()
    # Without a gap every loop comes back turned by a full turn: the rotor is -1, a Berry phase of pi.
    # With one, a small loop's phase is that of a cone, pi times one less the gap over the field's length
    # there, in both valleys. The rotor's scalar part is the cosine of the phase; comparing one minus it
    # keeps small phases visible.
    assert ((holonomies[gaps <= 0].select[0] + 1).abs() < 1e-9).all()
    cone_phases = np.pi * (1 - gaps / (gaps.squared() + (cone_slope * radii[0]).squared()).square_root())
    expected_drop = 1 - cone_phases.cos()[:, None]
    actual_drop = 1 - holonomies[:, :, 0].select[0]                             # [n_gaps, 2] Scalar+
    assert ((actual_drop - expected_drop).abs() <= 1e-2 * expected_drop).all()


if __name__ == "__main__":
    main()
