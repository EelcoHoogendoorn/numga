"""Scenes for graphene: the two bands over the Brillouin zone, the pseudospin around the two
valleys, and the Berry phase of loops around a valley, with and without a gap.

Momenta are in inverse carbon-carbon distances, 1 / 0.142 nm, and energies in eV.
"""

from __future__ import annotations

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
N_VALLEY_SAMPLES = 15
TEXTURE_GAP = 0.3
# The Berry phase: the loops' radii in inverse carbon-carbon distances, the gaps in eV, and the steps around.
RADII = np.linspace(0.01, 0.6, 30)
GAPS = np.array([0.0, 0.2, 0.5, 1.0])
N_LOOP_STEPS = 400


# --- math -----------------------------------------------------------------------------
def bands(extent: float, n_samples: int, gap: float):
    """The pseudospin field and the Hamiltonian's eigenvalues over a square of momenta that holds
    the Brillouin zone."""
    along = np.linspace(-extent, extent, n_samples)
    kx, ky = np.meshgrid(along, along)
    momenta = mv.vector(np.stack([kx, ky, np.zeros_like(kx)], axis=-1))          # [samples, samples] Vector
    field = core.pseudospin(momenta, gap)                            # [samples, samples] Vector
    values, _ = core.hamiltonian(field).eigh()                                    # [samples, samples, 4] Scalar
    return momenta, field, values


def textures(radius: float, n_samples: int, gap: float) -> tuple[core.Vector, core.Vector]:
    """The pseudospin of the upper band on a small square of momenta about each valley."""
    along = np.linspace(-radius, radius, n_samples)
    kx, ky = np.meshgrid(along, along)
    offsets = mv.vector(np.stack([kx, ky, np.zeros_like(kx)], axis=-1))           # [samples, samples] Vector
    field = core.pseudospin(core.VALLEYS[:, None, None] + offsets, gap)   # [valleys, samples, samples] Vector
    _, states = core.hamiltonian(field).eigh()                                    # [valleys, samples, samples, 4] Even
    directions = core.direction(states[..., -1])                                  # [valleys, samples, samples] Vector
    return offsets, directions


def berry(radii: np.ndarray, gaps: np.ndarray, n_steps: int) -> tuple[core.Rotor, core.Vector]:
    """The holonomy of loops of the given radii about each valley, for each gap, with the
    pseudospin direction each loop starts from."""
    loops = core.VALLEYS[:, None] + core.circle(n_steps)[:, None, None] * radii   # [steps + 1, valleys, radii] Vector
    field = core.pseudospin(loops[:, None], gaps[:, None, None])  # [steps + 1, gaps, valleys, radii] Vector
    directions = field.normalized()                                                # [steps + 1, gaps, valleys, radii] Vector
    rotors, starts = core.transport(directions)[-1], directions[0]                 # [gaps, valleys, radii] Rotor, Vector
    return rotors, starts


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_figure
    from examples.quantum.graphene import render

    momenta, field, values = bands(ZONE_EXTENT, N_ZONE_SAMPLES, GAPLESS)
    offsets, directions = textures(VALLEY_EXTENT, N_VALLEY_SAMPLES, TEXTURE_GAP)
    rotors, starts = berry(RADII, GAPS, N_LOOP_STEPS)
    save_figure(render.draw_field(momenta, field), "graphene_field")
    save_figure(render.draw_bands(momenta, values), "graphene_bands")
    save_figure(render.draw_textures(offsets, directions), "graphene_pseudospin")
    save_figure(render.draw_berry(RADII, GAPS, rotors, starts), "graphene_berry")

    # checks
    # The energies are minus and plus the field's length, each twice; without a gap the field
    # vanishes at the valleys, where the cones meet.
    length = (field | field).square_root().to_array()[..., None]
    np.testing.assert_allclose(values.to_array(), np.concatenate([-length, -length, length, length], -1), atol=1e-9)
    np.testing.assert_allclose(core.pseudospin(core.VALLEYS, 0.0).kernel, 0.0, atol=1e-6)
    # The upper band's pseudospin points along the field.
    valley_field = core.pseudospin(core.VALLEYS[:, None, None] + offsets, TEXTURE_GAP)   # [valleys, samples, samples] Vector
    np.testing.assert_allclose(directions.kernel, valley_field.normalized().kernel, atol=1e-10)
    # Without a gap every loop comes back turned by a full turn: the rotor is -1, a Berry phase of pi.
    # With one, a small loop's phase is that of a cone, pi times one less the gap over the field's length
    # there, in both valleys. The rotor's scalar part is the cosine of the phase; comparing one minus it
    # keeps small phases visible.
    np.testing.assert_allclose(rotors[GAPS == 0.0].select[0].to_array(), -1.0, atol=1e-12)
    near = np.pi * (1 - GAPS / np.sqrt(GAPS**2 + (VELOCITY * RADII[0]) ** 2))
    assert np.allclose(1 - rotors[:, :, 0].select[0].to_array(), 1 - np.cos(near)[:, None], rtol=1e-2, atol=0)


if __name__ == "__main__":
    main()
