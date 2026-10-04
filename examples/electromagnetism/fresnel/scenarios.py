"""Coincident, uniaxial and biaxial wave surfaces, computed from adjugates."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from examples.electromagnetism.fresnel import core

CASE_NAMES = ("Vacuum", "Uniaxial crystal", "Biaxial crystal")
PERMITTIVITY_X = np.array([1.0, 2.25, 1.44])
PERMITTIVITY_Y = np.array([1.0, 2.25, 2.25])
PERMITTIVITY_Z = np.array([1.0, 3.24, 3.24])
PERMITTIVITY = np.stack([PERMITTIVITY_X, PERMITTIVITY_Y, PERMITTIVITY_Z], axis=-1)
RELUCTIVITY = np.array([1.0, 1.0, 1.0])
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
    residual = core.polynomial(media[:, None, None, None], core.mv.t, wavevectors)
    wave = core.potential_map(media[:, None, None, None], wavevectors)
    np.testing.assert_allclose(residual.kernel, 0, atol=1e-9)
    np.testing.assert_allclose(wave(wavevectors).kernel, 0, atol=1e-11)
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
