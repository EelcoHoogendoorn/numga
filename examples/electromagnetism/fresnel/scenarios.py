"""Isotropic, uniaxial, and biaxial dielectrics illustrate static Fresnel wave surfaces and a continuous birefringence sweep."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import stack

from examples.electromagnetism.fresnel import core

# Principal permittivities along x, y, and z and scalar reluctivities for vacuum, uniaxial, and biaxial media.
case_permittivities = np.array([[1, 1, 1],
                                [2.25, 2.25, 3.24],
                                [1.44, 2.25, 3.24]])
case_reluctivities = np.array([1, 1, 1])
n_cases = len(case_permittivities)
axion_coupling = 0.3
frequency = 1.0
n_latitudes = 33
n_longitudes = 2 * (n_latitudes - 1) + 1
n_section_samples = 4 * (n_latitudes - 1)
n_frames = 72
n_movie_latitudes = 19
n_movie_longitudes = 2 * (n_movie_latitudes - 1) + 1
frame_duration_ms = 40


# --- math -----------------------------------------------------------------------------
def comparison_sheets() -> tuple[core.Constitutive, core.Vector, core.Vector, core.Vector]:
    """Principal permittivities across the `n_cases` media determine both 3D polarization sheets and their xz meridian sections."""
    media = core.dielectric(case_permittivities, case_reluctivities)               # [n_cases] Antibivector- <- Bivector+
    wavefronts = core.sphere(n_latitudes, n_longitudes)                            # [n_latitudes, n_longitudes] Vector+
    section_wavefronts = core.circle(n_section_samples)                            # [n_section_samples] Vector+
    sections = core.radial_sheets(media[:, None], section_wavefronts, frequency)   # [n_cases, n_section_samples, 2] Vector+
    sheets = core.radial_sheets(media[:, None, None], wavefronts, frequency)       # [n_cases, n_latitudes, n_longitudes, 2] Vector+
    return media, wavefronts, sheets, sections


def birefringence_states() -> Iterator[tuple[core.Vector, core.Vector]]:
    """Yields `(sheets, sections)` states as the dielectric interpolates smoothly from vacuum to a biaxial crystal."""
    wavefronts = core.sphere(n_movie_latitudes, n_movie_longitudes)                # [n_movie_latitudes, n_movie_longitudes] Vector+
    section_wavefronts = core.circle(n_section_samples)                            # [n_section_samples] Vector+
    # Ease smoothly from vacuum to the biaxial crystal.
    weights = (1 - np.cos(np.linspace(0, np.pi, n_frames // 2 + 1))) / 2
    for weight in weights:
        permittivity = case_permittivities[0] + weight * (case_permittivities[-1] - case_permittivities[0])
        medium = core.dielectric(permittivity, case_reluctivities[-1])             # [] Antibivector- <- Bivector+
        sheets = core.radial_sheets(medium, wavefronts, frequency)                 # [n_movie_latitudes, n_movie_longitudes, 2] Vector+
        sections = core.radial_sheets(medium, section_wavefronts, frequency)       # [n_section_samples, 2] Vector+
        yield sheets, sections


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.electromagnetism.fresnel import render
    from examples.util.animation import save_animation, save_figure

    media, wavefronts, sheets, sections = comparison_sheets()
    save_figure(render.comparison_figure(sheets, sections), "fresnel_adjugate")
    streamed_states = list(birefringence_states())
    save_animation(render.sheet_animation(streamed_states), "fresnel_adjugate", frame_duration_ms, forward_then_back=True)

    # checks
    potentials = stack((core.mv.x, core.mv.y, core.mv.z, core.mv.t))               # [4] Vector+
    wavevectors = core.mv.t * frequency + sheets                                   # [n_cases, n_latitudes, n_longitudes, 2] Vector+
    residuals = core.fresnel_quartic(media[:, None, None, None], core.time, core.rest_space, wavevectors)  # [n_cases, n_latitudes, n_longitudes, 2] Scalar+
    waves = core.potential_map(media[:, None, None, None], wavevectors)            # [n_cases, n_latitudes, n_longitudes, 2] Antivector- <- Vector+
    assert (residuals.abs() < 1e-9).all()
    assert ((potentials & waves(wavevectors)[..., None]).abs() < 1e-11).all()
    assert ((streamed_states[0][1] - sections[0]).norm() < 1e-6).all()
    assert ((streamed_states[-1][1] - sections[-1]).norm() < 1e-6).all()


if __name__ == "__main__":
    main()
