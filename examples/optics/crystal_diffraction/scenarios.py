"""A finite crystal and its scattering pattern under stretch, shear and rotation."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from examples.optics.crystal_diffraction import core

N_SHELLS = 3
N_ORDERS = 3
SPACING = 1.0
MAX_STRETCH = 1.7
MAX_SHEAR = 0.6
MAX_ANGLE = np.pi / 9
# The cases: the square layer, stretched, and stretched, sheared and turned.
STRETCHES = np.array([1.0, MAX_STRETCH, MAX_STRETCH])
SHEARS = np.array([0.0, 0.0, MAX_SHEAR])
ANGLES = np.array([0.0, 0.0, MAX_ANGLE])
HALF_WIDTH = 10.0
N_PIXELS = 161
N_FRAMES = 20
N_MOVIE_PIXELS = 81
FRAME_DURATION = 120


# --- math -----------------------------------------------------------------------------
def comparison() -> tuple[core.Crystal, core.Crystal, core.Plane, core.Scalar]:
    """The square layer at rest, the three crystals carried from it, the transfer grid, and their coherent
    scattering intensities."""
    reference = core.square(N_SHELLS, N_ORDERS, SPACING)
    crystals = reference.carried(core.deformation(STRETCHES, SHEARS, ANGLES))
    grid = core.transfer_grid(HALF_WIDTH, N_PIXELS)
    return reference, crystals, grid, crystals.intensity(grid)


def deformation(grid: core.Plane) -> Iterator[tuple[core.Point, core.Plane, core.Scalar]]:
    """Per frame of a stretch, shear and turn, the atoms, the reciprocal lattice and the intensity; the animation
    plays them forward and back."""
    reference = core.square(N_SHELLS, N_ORDERS, SPACING)
    for phase in np.linspace(0, np.pi, N_FRAMES // 2 + 1):
        weight = np.array((1 - np.cos(phase)) / 2)
        crystal = reference.carried(core.deformation(1 + (MAX_STRETCH - 1) * weight, MAX_SHEAR * weight, MAX_ANGLE * weight))
        yield crystal.positions, crystal.reciprocal, crystal.intensity(grid)


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.optics.crystal_diffraction import render

    reference, crystals, grid, intensity = comparison()
    movie_grid = core.transfer_grid(HALF_WIDTH, N_MOVIE_PIXELS)
    save_figure(render.comparison(crystals.positions, crystals.reciprocal, grid, intensity), "crystal_diffraction")
    save_animation(render.animate(deformation(movie_grid), movie_grid), "crystal_diffraction", FRAME_DURATION,
                   forward_then_back=True)

    # checks
    # Every phase measured at every atom is kept.
    phases = crystals.reciprocal[:, :, None] & crystals.positions[:, None, :]
    reference_phases = reference.reciprocal[:, None] & reference.positions[None, :]
    np.testing.assert_allclose((phases - reference_phases).kernel, 0, atol=1e-10)


if __name__ == "__main__":
    main()
