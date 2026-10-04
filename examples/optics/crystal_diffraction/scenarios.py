"""A finite crystal and its scattering pattern under stretch, shear and rotation."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from examples.optics.crystal_diffraction import core

SHELLS = 3
ORDERS = 3
SPACING = 1.0
CASE_NAMES = ("Square lattice", "Stretched", "Stretched, sheared and turned")
MAX_STRETCH = 1.7
MAX_SHEAR = 0.6
MAX_ANGLE = np.pi / 9
STRETCHES = np.array([1.0, MAX_STRETCH, MAX_STRETCH])
SHEARS = np.array([0.0, 0.0, MAX_SHEAR])
ANGLES = np.array([0.0, 0.0, MAX_ANGLE])
HALF_WIDTH = 10.0
PIXELS = 161
FRAMES = 20
MOVIE_PIXELS = 81
DURATION_MS = 120


# --- math -----------------------------------------------------------------------------
def comparison() -> tuple[core.Crystal, core.Vector, core.Scalar]:
    """The three crystals, the transfer grid and their coherent scattering intensities."""
    reference = core.square(SHELLS, ORDERS, SPACING)
    deformations = core.deformation(STRETCHES, SHEARS, ANGLES)
    crystals = reference.carried(deformations)
    grid = core.transfer_grid(HALF_WIDTH, PIXELS)

    # --- checks
    # Every phase measured at every atom is kept, and the phase planes are the reciprocal vectors' duals.
    phases = crystals.reciprocal[:, :, None] | crystals.positions[:, None, :]
    reference_phases = reference.reciprocal[:, None] | reference.positions[None, :]
    np.testing.assert_allclose((phases - reference_phases).kernel, 0, atol=1e-10)
    np.testing.assert_allclose((crystals.phase_planes - crystals.reciprocal.dual()).kernel, 0, atol=1e-11)
    return crystals, grid, crystals.intensity(grid)


def deformation(grid: core.Vector) -> Iterator[tuple[core.Vector, core.Vector, core.Scalar]]:
    """The atoms, reciprocal vectors and intensity through a stretch, shear and turn."""
    reference = core.square(SHELLS, ORDERS, SPACING)
    phase = np.linspace(0, 2 * np.pi, FRAMES, endpoint=False)
    weight = (1 - np.cos(phase)) / 2
    maps = core.deformation(1 + (MAX_STRETCH - 1) * weight, MAX_SHEAR * weight, MAX_ANGLE * weight)
    crystals = reference.carried(maps)
    intensity = crystals.intensity(grid)
    yield from zip(crystals.positions, crystals.reciprocal, intensity)


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.animation import save_animation, save_figure
    from examples.optics.crystal_diffraction import render

    crystals, grid, intensity = comparison()
    movie_grid = core.transfer_grid(HALF_WIDTH, MOVIE_PIXELS)
    save_figure(render.comparison(crystals, grid, intensity, CASE_NAMES), "crystal_diffraction")
    save_animation(render.animate(deformation(movie_grid), movie_grid), "crystal_diffraction", DURATION_MS)


if __name__ == "__main__":
    main()
