"""A finite crystal and the light it scatters under stretch, shear and rotation."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from examples.optics.crystal_diffraction import core

n_shells = 3
n_orders = 3
spacing = 1.0
max_stretch = 1.7
max_shear = 0.6
max_angle = np.pi / 9
# The cases: the square layer, stretched, and stretched, sheared and turned.
stretches = np.array([1.0, max_stretch, max_stretch])
shears = np.array([0.0, 0.0, max_shear])
angles = np.array([0.0, 0.0, max_angle])
half_width = 10.0
n_pixels = 101
n_frames = 60
n_movie_pixels = 211
frame_duration = 40
# Light running along z: its phase stays put along `time + mv.z.dual()`. Its electric field lies along x.
wavevector = core.mv.t - core.mv.z                                          # [] Vector+
incident = wavevector ^ core.mv.x                                           # [] Bivector+


# --- math -----------------------------------------------------------------------------
def comparison() -> tuple[core.Crystal, core.Crystal, core.Vector, core.Scalar]:
    """The square layer at rest, the three crystals carried from it, the transfer grid, and the energy density of the
    light each crystal scatters to it."""
    reference = core.square(n_shells, n_orders, spacing)
    crystals = reference.carried(core.deformation(stretches, shears, angles))
    grid = core.transfer_grid(half_width, n_pixels)
    field = crystals.scattered(grid, incident)                             # [cases, rows, columns] Bivector+
    return reference, crystals, grid, core.energy_density(field, field)


def deformation(grid: core.Vector) -> Iterator[tuple[core.Antivector, core.Vector, core.Scalar]]:
    """Per frame of a stretch, shear and turn, the atoms, the reciprocal lattice and the energy density of the
    scattered light; the animation plays them forward and back."""
    reference = core.square(n_shells, n_orders, spacing)
    for phase in np.linspace(0, np.pi, n_frames // 2 + 1):
        weight = np.array((1 - np.cos(phase)) / 2)
        crystal = reference.carried(core.deformation(1 + (max_stretch - 1) * weight, max_shear * weight,
                                                     max_angle * weight))
        field = crystal.scattered(grid, incident)                          # [rows, columns] Bivector+
        yield crystal.positions, crystal.reciprocal, core.energy_density(field, field)


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.optics.crystal_diffraction import render

    reference, crystals, grid, energy_density = comparison()
    movie_grid = core.transfer_grid(half_width, n_movie_pixels)
    save_figure(render.comparison(crystals.positions, crystals.reciprocal, grid, energy_density),
                "crystal_diffraction")
    save_animation(render.animate(deformation(movie_grid), movie_grid), "crystal_diffraction", frame_duration,
                   forward_then_back=True)

    # checks
    # Every phase measured at every atom is kept.
    phases = crystals.reciprocal[:, :, None] & crystals.positions[:, None, :]
    reference_phases = reference.reciprocal[:, None] & reference.positions[None, :]
    np.testing.assert_allclose((phases - reference_phases).kernel, 0, atol=1e-10)


if __name__ == "__main__":
    main()
