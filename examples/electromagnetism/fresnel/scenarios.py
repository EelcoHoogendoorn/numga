"""Fresnel wave surfaces: coincident, uniaxial and biaxial, computed from adjugates."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from examples.electromagnetism.fresnel import core

# One row per case: its permittivity along the principal axes x, y and z, and its reluctivity.
case_permittivities = np.array([[1, 1, 1],
                                [2.25, 2.25, 3.24],
                                [1.44, 2.25, 3.24]])
case_reluctivities = np.array([1, 1, 1])
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
def comparison() -> tuple[core.Constitutive, core.Vector, core.Vector, core.Vector]:
    """The media, the wavefronts, and both wavevector sheets and their xz sections for each material."""
    media = core.dielectric(case_permittivities, case_reluctivities)               # [cases] Antibivector- <- Bivector+
    wavefronts = core.sphere(n_latitudes, n_longitudes)                            # [latitudes, longitudes] Vector+
    section_wavefronts = core.circle(n_section_samples)                            # [section_samples] Vector+
    sections = core.radial_sheets(media[:, None], section_wavefronts, frequency)   # [cases, section_samples, sheets] Vector+
    sheets = core.radial_sheets(media[:, None, None], wavefronts, frequency)       # [cases, latitudes, longitudes, sheets] Vector+

    return media, wavefronts, sheets, sections


def birefringence() -> Iterator[tuple[core.Vector, core.Vector]]:
    """The sheets and sections as the dielectric changes from vacuum to a biaxial crystal, frame by frame; the
    animation plays the change forward and back."""
    wavefronts = core.sphere(n_movie_latitudes, n_movie_longitudes)                # [latitudes, longitudes] Vector+
    section_wavefronts = core.circle(n_section_samples)                            # [section_samples] Vector+
    # An easing from vacuum to the crystal, slow at either end.
    weights = (1 - np.cos(np.linspace(0, np.pi, n_frames // 2 + 1))) / 2
    for weight in weights:
        permittivity = case_permittivities[0] + weight * (case_permittivities[-1] - case_permittivities[0])
        medium = core.dielectric(permittivity, case_reluctivities[-1])             # [] Antibivector- <- Bivector+
        sheets = core.radial_sheets(medium, wavefronts, frequency)                 # [latitudes, longitudes, sheets] Vector+
        sections = core.radial_sheets(medium, section_wavefronts, frequency)       # [section_samples, sheets] Vector+
        yield sheets, sections


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.electromagnetism.fresnel import render

    media, wavefronts, sheets, sections = comparison()
    save_figure(render.comparison(sheets, sections), "fresnel_adjugate")
    save_animation(render.animate(birefringence()), "fresnel_adjugate", frame_duration, forward_then_back=True)

    # checks
    # Every wavevector on the sheets solves the quartic, and lies in its potential map's kernel.
    wavevectors = core.mv.t * frequency + sheets
    residual = core.polynomial(media[:, None, None, None], core.time, core.rest_space, wavevectors)
    wave = core.potential_map(media[:, None, None, None], wavevectors)
    np.testing.assert_allclose(residual.kernel, 0, atol=1e-9)
    np.testing.assert_allclose(wave(wavevectors).kernel, 0, atol=1e-11)
    # The axion part adds nothing to any potential map, and leaves every sheet where it is; where two
    # sheets coincide, the square root of the vanishing discriminant loses half the digits.
    axion = core.mv(core.AxionCoupling, [axion_coupling]) * core.Bivector          # [] Antibivector- <- Bivector+
    np.testing.assert_allclose(core.potential_map(axion, wavevectors).kernel, 0, atol=1e-12)
    coupled_sheets = core.radial_sheets((media + axion)[:, None, None], wavefronts, frequency)
    np.testing.assert_allclose((coupled_sheets - sheets).kernel, 0, atol=1e-6)


if __name__ == "__main__":
    main()
