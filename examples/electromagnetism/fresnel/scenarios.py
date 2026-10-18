"""Coincident, uniaxial and biaxial wave surfaces, computed from adjugates."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from examples.electromagnetism.fresnel import core

CASE_NAMES = ("Vacuum", "Uniaxial crystal", "Biaxial crystal")
# One row per case: its permittivity along the principal axes x, y and z, and its reluctivity.
PERMITTIVITY = np.array([[1.0, 1.0, 1.0],
                         [2.25, 2.25, 3.24],
                         [1.44, 2.25, 3.24]])
RELUCTIVITY = np.array([1.0, 1.0, 1.0])
AXION_COUPLING = 0.3                                                              # illustrative
FREQUENCY = 1.0
LATITUDES = 33
LONGITUDES = 2 * (LATITUDES - 1) + 1
SECTION_SAMPLES = 4 * (LATITUDES - 1) + 1
FRAMES = 32
MOVIE_LATITUDES = 19
MOVIE_LONGITUDES = 2 * (MOVIE_LATITUDES - 1) + 1
DURATION_MS = 90


# --- math -----------------------------------------------------------------------------
def comparison() -> tuple[core.Vector, core.Vector]:
    """Both wavevector sheets and their xz sections for each material."""
    media = core.dielectric(PERMITTIVITY, RELUCTIVITY)
    directions = core.sphere(LATITUDES, LONGITUDES)
    sections = core.radial_sheets(media[:, None], core.circle(SECTION_SAMPLES), FREQUENCY)
    sheets = core.radial_sheets(media[:, None, None], directions, FREQUENCY)

    # --- checks
    # Every wavevector on the sheets solves the quartic, and lies in its potential map's kernel.
    wavevectors = core.mv.t * FREQUENCY + sheets
    residual = core.polynomial(media[:, None, None, None], core.mv.xyz, wavevectors)
    wave = core.potential_map(media[:, None, None, None], wavevectors)
    np.testing.assert_allclose(residual.kernel, 0, atol=1e-9)
    np.testing.assert_allclose(wave(wavevectors).kernel, 0, atol=1e-11)
    # The axion part adds nothing to any potential map, and leaves every sheet where it is; where two
    # sheets coincide, the square root of the vanishing discriminant loses half the digits.
    axion = core.axion(core.mv(core.Axion, [AXION_COUPLING]))                      # [] Antibivector <- Bivector
    np.testing.assert_allclose(core.potential_map(axion, wavevectors).kernel, 0, atol=1e-12)
    coupled_sheets = core.radial_sheets((media + axion)[:, None, None], directions, FREQUENCY)
    np.testing.assert_allclose((coupled_sheets - sheets).kernel, 0, atol=1e-6)
    return sheets, sections


def birefringence() -> Iterator[tuple[core.Vector, core.Vector]]:
    """The sheets and sections as the dielectric changes from vacuum to a biaxial crystal."""
    phase = np.linspace(0, 2 * np.pi, FRAMES, endpoint=False)
    weight = (1 - np.cos(phase)) / 2
    permittivity = PERMITTIVITY[0] + weight[:, None] * (PERMITTIVITY[-1] - PERMITTIVITY[0])
    media = core.dielectric(permittivity, RELUCTIVITY[-1])
    directions = core.sphere(MOVIE_LATITUDES, MOVIE_LONGITUDES)
    sheets = core.radial_sheets(media[:, None, None], directions, FREQUENCY)
    sections = core.radial_sheets(media[:, None], core.circle(SECTION_SAMPLES), FREQUENCY)
    yield from zip(sheets, sections)


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.animation import save_animation, save_figure
    from examples.electromagnetism.fresnel import render

    sheets, sections = comparison()
    save_figure(render.comparison(sheets, sections, CASE_NAMES), "fresnel_adjugate")
    save_animation(render.animate(birefringence()), "fresnel_adjugate", DURATION_MS)


if __name__ == "__main__":
    main()
