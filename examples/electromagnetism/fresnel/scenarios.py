"""Fresnel wave surfaces: coincident, uniaxial and biaxial, computed from adjugates."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from examples.electromagnetism.fresnel import core

case_names = ("Vacuum", "Uniaxial crystal", "Biaxial crystal")
# One row per case: its permittivity along the principal axes x, y and z, and its reluctivity.
case_permittivities = np.array([[1.0, 1.0, 1.0],
                         [2.25, 2.25, 3.24],
                         [1.44, 2.25, 3.24]])
case_reluctivities = np.array([1.0, 1.0, 1.0])
# An axion coupling, of an illustrative size: light does not see it.
axion_coupling = 0.3
frequency = 1.0
n_latitudes = 33
n_longitudes = 2 * (n_latitudes - 1) + 1
n_section_samples = 4 * (n_latitudes - 1) + 1
n_frames = 32
n_movie_latitudes = 19
n_movie_longitudes = 2 * (n_movie_latitudes - 1) + 1
frame_duration = 90


# --- math -----------------------------------------------------------------------------
def comparison() -> tuple[core.Vector, core.Vector]:
    """Both wavevector sheets and their xz sections for each material, with the media and the directions."""
    media = core.dielectric(case_permittivities, case_reluctivities)
    directions = core.sphere(n_latitudes, n_longitudes)
    sections = core.radial_sheets(media[:, None], core.circle(n_section_samples), frequency)
    sheets = core.radial_sheets(media[:, None, None], directions, frequency)

    return media, directions, sheets, sections


def birefringence() -> Iterator[tuple[core.Vector, core.Vector]]:
    """The sheets and sections as the dielectric changes from vacuum to a biaxial crystal."""
    phase = np.linspace(0, 2 * np.pi, n_frames, endpoint=False)
    weight = (1 - np.cos(phase)) / 2
    permittivity = case_permittivities[0] + weight[:, None] * (case_permittivities[-1] - case_permittivities[0])
    media = core.dielectric(permittivity, case_reluctivities[-1])
    directions = core.sphere(n_movie_latitudes, n_movie_longitudes)
    sheets = core.radial_sheets(media[:, None, None], directions, frequency)
    sections = core.radial_sheets(media[:, None], core.circle(n_section_samples), frequency)
    yield from zip(sheets, sections)


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.animation import save_animation, save_figure
    from examples.electromagnetism.fresnel import render

    media, directions, sheets, sections = comparison()
    save_figure(render.comparison(sheets, sections, case_names), "fresnel_adjugate")
    save_animation(render.animate(birefringence()), "fresnel_adjugate", frame_duration)

    # checks
    # Every wavevector on the sheets solves the quartic, and lies in its potential map's kernel.
    wavevectors = core.mv.t * frequency + sheets
    residual = core.polynomial(media[:, None, None, None], core.mv.xyz, wavevectors)
    wave = core.potential_map(media[:, None, None, None], wavevectors)
    np.testing.assert_allclose(residual.kernel, 0, atol=1e-9)
    np.testing.assert_allclose(wave(wavevectors).kernel, 0, atol=1e-11)
    # The axion part adds nothing to any potential map, and leaves every sheet where it is; where two
    # sheets coincide, the square root of the vanishing discriminant loses half the digits.
    axion = core.axion(core.mv(core.Axion, [axion_coupling]))                      # [] Antibivector <- Bivector
    np.testing.assert_allclose(core.potential_map(axion, wavevectors).kernel, 0, atol=1e-12)
    coupled_sheets = core.radial_sheets((media + axion)[:, None, None], directions, frequency)
    np.testing.assert_allclose((coupled_sheets - sheets).kernel, 0, atol=1e-6)


if __name__ == "__main__":
    main()
