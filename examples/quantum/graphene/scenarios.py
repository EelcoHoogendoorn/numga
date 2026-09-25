"""Scenes for graphene: the two bands over the Brillouin zone, the pseudospin around the two
valleys, and the Berry phase of loops around a valley, with and without a gap.

Momenta are in inverse carbon-carbon distances, 1 / 0.142 nm, and energies in eV.
"""

from __future__ import annotations

import numpy as np

from examples.quantum.graphene import core

mv = core.mv
# The slope of the cones.
VELOCITY = 1.5 * core.HOPPING                       # eV per inverse carbon-carbon distance


# --- math -----------------------------------------------------------------------------
def bands(extent: float, count: int, gap: float):
    """The pseudospin field and the Hamiltonian's eigenvalues over a square of momenta that holds
    the Brillouin zone."""
    along = np.linspace(-extent, extent, count)
    kx, ky = np.meshgrid(along, along)
    momenta = mv.vector(np.stack([kx, ky, np.zeros_like(kx)], axis=-1))          # [count, count] Vector
    field = core.pseudospin(momenta, mv.scalar([gap]))                            # [count, count] Vector
    values, _ = core.hamiltonian(field).eigh()                                    # [count, count, 4] Scalar

    # --- checks
    # The energies are minus and plus the field's length, each twice; without a gap the field
    # vanishes at the valleys, where the cones meet.
    length = (field | field).square_root().to_array()[..., None]
    np.testing.assert_allclose(values.to_array(), np.concatenate([-length, -length, length, length], -1), atol=1e-9)
    np.testing.assert_allclose(core.pseudospin(core.VALLEYS, mv.scalar([0.0])).kernel, 0.0, atol=1e-6)
    return momenta, field, values


def textures(radius: float, count: int, gap: float) -> tuple[core.Vector, core.Vector]:
    """The pseudospin of the upper band on a small square of momenta about each valley."""
    along = np.linspace(-radius, radius, count)
    kx, ky = np.meshgrid(along, along)
    offsets = mv.vector(np.stack([kx, ky, np.zeros_like(kx)], axis=-1))           # [count, count] Vector
    field = core.pseudospin(core.VALLEYS[:, None, None] + offsets, mv.scalar([gap]))   # [valleys, count, count] Vector
    _, states = core.hamiltonian(field).eigh()                                    # [valleys, count, count, 4] Even
    directions = core.direction(states[..., -1])                                  # [valleys, count, count] Vector

    # --- checks
    # The upper band's pseudospin points along the field.
    np.testing.assert_allclose(directions.kernel, field.normalized().kernel, atol=1e-10)
    return offsets, directions


def berry(radii: np.ndarray, gaps: np.ndarray, count: int) -> tuple[core.Rotor, core.Vector]:
    """The holonomy of loops of the given radii about each valley, for each gap, with the
    pseudospin direction each loop starts from."""
    loops = core.VALLEYS[:, None] + core.circle(count)[:, None, None] * mv.scalar(radii[:, None])   # [count + 1, valleys, radii] Vector
    field = core.pseudospin(loops[:, None], mv.scalar(gaps[:, None, None, None]))  # [count + 1, gaps, valleys, radii] Vector
    directions = field.normalized()                                                # [count + 1, gaps, valleys, radii] Vector
    rotors, starts = core.transport(directions)[-1], directions[0]                 # [gaps, valleys, radii] Rotor, Vector

    # --- checks
    # Without a gap every loop comes back turned by a full turn: the rotor is -1, a Berry phase of pi.
    # With one, a small loop's phase is that of a cone,
    # `np.pi * (1 - gaps / np.sqrt(gaps ** 2 + (VELOCITY * radii) ** 2))`, in both valleys. The
    # rotor's scalar part is the cosine of the phase; comparing one minus it keeps small phases visible.
    np.testing.assert_allclose(rotors[gaps == 0.0].select[0].to_array(), -1.0, atol=1e-12)
    near = np.pi * (1 - gaps / np.sqrt(gaps**2 + (VELOCITY * radii[0]) ** 2))
    assert np.allclose(1 - rotors[:, :, 0].select[0].to_array(), 1 - np.cos(near)[:, None], rtol=1e-2, atol=0)
    return rotors, starts


if __name__ == "__main__":
    from examples.animation import save_figure
    from examples.quantum.graphene import render

    momenta, field, values = bands(4.5, 121, 0.0)
    save_figure(render.draw_field(momenta, field), "graphene_field")
    save_figure(render.draw_bands(momenta, values), "graphene_bands")
    save_figure(render.draw_textures(*textures(0.5, 15, 0.3)), "graphene_pseudospin")
    radii, gaps = np.linspace(0.01, 0.6, 30), np.array([0.0, 0.2, 0.5, 1.0])
    save_figure(render.draw_berry(radii, gaps, *berry(radii, gaps, 400)), "graphene_berry")
