"""Scenes for graphene: the two bands over the Brillouin zone, the pseudospin around the two
valleys, and the Berry phase of loops around a valley, with and without a gap.

Momenta are in inverse carbon-carbon distances, 1 / 0.142 nm, and energies in eV.
"""

from __future__ import annotations

from functools import partial

import numpy as np

from examples.quantum.graphene import core

mv = core.mv
# The slope of the cones, in eV per inverse carbon-carbon distance.
VELOCITY = 1.5 * core.HOPPING
# The bands: the half width of the square of momenta that holds the Brillouin zone, its samples each way,
# and no gap.
ZONE_EXTENT = 4.5
N_ZONE_SAMPLES = 121
GAPLESS = 0.0
# The pseudospin about the valleys: the half width of the square about each, its samples each way, and the gap.
VALLEY_EXTENT = 0.5
N_VALLEY_SAMPLES = 11
TEXTURE_GAP = 0.3
# The Berry phase: the loops' radii in inverse carbon-carbon distances, the gaps in eV, and the steps around.
RADII = np.linspace(0.01, 0.6, 30)
GAPS = np.array([0.0, 0.2, 0.5, 1.0])
N_LOOP_STEPS = 400


# --- math -----------------------------------------------------------------------------
def bands(extent: float, n_samples: int, gap: float) -> tuple[core.Wavevector, core.Field, core.Scalar]:
    """The pseudospin field and the Hamiltonian's eigenvalues over a square of momenta that holds
    the Brillouin zone."""
    along = np.linspace(-extent, extent, n_samples)
    kx, ky = np.meshgrid(along, along)
    momenta = mv.x * kx + mv.y * ky                                             # [samples, samples] Wavevector+
    field = core.pseudospin_field(momenta, gap)                                 # [samples, samples] Field+
    energies, _ = core.hamiltonian(field).eigh()                                # [samples, samples, modes] Scalar+
    return momenta, field, energies


def upper_pseudospins(offsets: core.Wavevector, gap: float) -> core.Pseudospin:
    """The pseudospin of the upper band at momenta offset from each valley."""
    field = core.pseudospin_field(core.valleys[:, None, None] + offsets, gap)   # [valleys, samples, samples] Field+
    _, states = core.hamiltonian(field).eigh()                                  # [valleys, samples, samples, modes] Spinor+
    upper = states[..., -1]                                                     # [valleys, samples, samples] Spinor+
    # The pseudospin of a state: its sandwich of the first sublattice's pseudospin, over its norm.
    return (upper >> core.first_sublattice) / upper.symmetric_reverse_product()   # [valleys, samples, samples] Pseudospin-


def textures(extent: float, n_samples: int, gap: float) -> tuple[core.Wavevector, core.Pseudospin]:
    """The pseudospin of the upper band on a small square of momenta about each valley."""
    along = np.linspace(-extent, extent, n_samples)
    kx, ky = np.meshgrid(along, along)
    offsets = mv.x * kx + mv.y * ky                                             # [samples, samples] Wavevector+
    return offsets, upper_pseudospins(offsets, gap)


def berry(radii: np.ndarray, gaps: np.ndarray, n_steps: int) -> tuple[core.Rotor, core.Pseudospin]:
    """The holonomy of loops of the given radii about each valley, for each gap, with the
    pseudospin direction each loop starts from."""
    loops = core.valleys[:, None] + core.circle(n_steps)[:, None, None] * radii   # [steps + 1, valleys, radii] Wavevector+
    field = core.pseudospin_field(loops[:, None], gaps[:, None, None])  # [steps + 1, gaps, valleys, radii] Field+
    # The upper band's pseudospin, along the field.
    directions = field.normalized().dual()                              # [steps + 1, gaps, valleys, radii] Pseudospin-
    return core.transport(directions)[-1], directions[0]


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_figure
    from examples.quantum.graphene import render

    momenta, field, energies = bands(ZONE_EXTENT, N_ZONE_SAMPLES, GAPLESS)
    offsets, directions = textures(VALLEY_EXTENT, N_VALLEY_SAMPLES, TEXTURE_GAP)
    holonomies, starts = berry(RADII, GAPS, N_LOOP_STEPS)
    save_figure(render.draw_field(momenta, field), "graphene_field")
    save_figure(render.draw_bands(momenta, energies), "graphene_bands")
    save_figure(render.draw_textures(partial(upper_pseudospins, gap=TEXTURE_GAP), VALLEY_EXTENT), "graphene_pseudospin")
    save_figure(render.draw_berry(RADII, GAPS, holonomies, starts), "graphene_berry")

    # checks
    # The energies are minus and plus the field's length, each twice; without a gap the field
    # vanishes at the valleys, where the cones meet.
    length = field.norm().to_array()[..., None]
    np.testing.assert_allclose(energies.to_array(), np.concatenate([-length, -length, length, length], -1), atol=1e-9)
    np.testing.assert_allclose(core.pseudospin_field(core.valleys, 0.0).kernel, 0.0, atol=1e-6)
    # The upper band's pseudospin points along the field.
    valley_field = core.pseudospin_field(core.valleys[:, None, None] + offsets, TEXTURE_GAP)   # [valleys, samples, samples] Field+
    np.testing.assert_allclose(directions.kernel, valley_field.normalized().dual().kernel, atol=1e-10)
    # Without a gap every loop comes back turned by a full turn: the rotor is -1, a Berry phase of pi.
    # With one, a small loop's phase is that of a cone, pi times one less the gap over the field's length
    # there, in both valleys. The rotor's scalar part is the cosine of the phase; comparing one minus it
    # keeps small phases visible.
    np.testing.assert_allclose(holonomies[GAPS == 0.0].select[0].to_array(), -1.0, atol=1e-12)
    near = np.pi * (1 - GAPS / np.sqrt(GAPS**2 + (VELOCITY * RADII[0]) ** 2))
    assert np.allclose(1 - holonomies[:, :, 0].select[0].to_array(), 1 - np.cos(near)[:, None], rtol=1e-2, atol=0)


if __name__ == "__main__":
    main()
