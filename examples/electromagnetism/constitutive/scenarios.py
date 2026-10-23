"""Glass, a crystal and a ferrite: the speeds of the waves each carries along z and by heading, the fields
those waves carry, and glass in motion."""

from __future__ import annotations

import numpy as np

from numga import stack

from examples.electromagnetism.constitutive import core

mv, t, z, Bivector = core.mv, core.t, core.z, core.Bivector
Modes = tuple[np.ndarray, core.Bivector]

# The media at rest, one entry each for glass, a crystal and a ferrite: the permittivity along x, y and z,
# and the inverse permeability across each.
permittivities = np.array([[2.25, 2.25, 2.25], [2.25, 1.5, 1.5], [2.25, 2.25, 2.25]])
inverse_permeabilities = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 0.5, 1.0]])
# The glass's axion coupling, and the speed of the glass in motion as a fraction of the speed of light.
axion_coupling = 0.4
glass_speed = 0.35
# The trial speeds of waves along z, and the residual under which a local minimum counts as a wave.
speed_range = (0.05, 1.5)
n_speeds = 3001
threshold = 4e-3
# The headings through the xz plane, and the trial speeds along each.
n_headings = 72
heading_speed_range = (0.4, 1.0)
n_heading_speeds = 601
# Light with the glass and against it.
senses = np.array([1.0, -1.0])
# The speeds of the glass for the drag, the trial speeds of light through it, and the samples of the
# analytic curves.
drag_range = (-0.6, 0.6)
n_drags = 15
drag_speed_range = (0.05, 1.2)
n_drag_speeds = 1151
n_curve = 200
# The frames of one period of the waves, and the duration of each in milliseconds.
n_frames = 24
frame_duration = 50


# --- math -----------------------------------------------------------------------------
def media() -> core.Constitutive:
    """Glass, the crystal and the ferrite at rest `[media]`."""
    return core.medium(permittivities, inverse_permeabilities)


def moving(medium: core.Constitutive, fractions: np.ndarray) -> core.Constitutive:
    """The medium seen moving along z at each fraction of the speed of light: its rest map carried by the
    boost."""
    boosts = ((z ^ t) * (np.arctanh(fractions) / 2)).exp()                    # [...] Rotor
    return boosts >> medium(boosts << Bivector)                                # [...] Constitutive


def expected_speeds() -> np.ndarray:
    """The speeds of the two waves each medium carries along z `[media, 2]`: the field along x feels the
    permittivity along x and the inverse permeability across y, the field along y the other two."""
    return np.sqrt(inverse_permeabilities[:, [1, 0]] / permittivities[:, [0, 1]])


def relativistic_sum(fractions: np.ndarray) -> np.ndarray:
    """The speed of light through glass moving at each fraction of the speed of light, with the glass and
    against it `[2, ...]`."""
    light = expected_speeds()[0, 0]
    flow = senses[:, None] * fractions
    return (light + flow) / (1 + light * flow)


def dispersion() -> tuple[np.ndarray, core.Scalar, np.ndarray]:
    """The trial speeds along z, each medium's residual at each `[speeds, media]`, and the expected
    speeds `[media, 2]`."""
    speeds = np.linspace(*speed_range, n_speeds)
    return speeds, core.residuals((speeds * t - z)[:, None], media()), expected_speeds()


def modes() -> tuple[Modes, Modes]:
    """The fields the glass and the crystal carry along z, with their speeds: the glass two at one speed,
    the last two fields of its wave map, and the crystal one at each of its two speeds."""
    glass, crystal, _ = media()
    glass_speeds, crystal_speeds, _ = expected_speeds()
    _, _, glass_fields = core.wave_map(glass_speeds[0] * t - z, glass).svd()
    return (glass_speeds, glass_fields[-2:]), (crystal_speeds, core.field_modes(crystal_speeds * t - z, crystal))


def headings() -> tuple[np.ndarray, core.Scalar, np.ndarray]:
    """The headings through the xz plane, the residual at every trial speed and heading for the glass, the
    crystal and the glass in motion `[speeds, headings, media]`, and the speeds."""
    angles = np.linspace(0, 2 * np.pi, n_headings)
    speeds = np.linspace(*heading_speed_range, n_heading_speeds)
    # z turned towards x by each angle.
    directions = (mv.zx * (-angles / 2)).exp() >> z                           # [headings] Vector
    glass, crystal, _ = media()
    cases = stack([glass, crystal, moving(glass, np.array(glass_speed))])    # [media] Constitutive
    wavevectors = speeds[:, None] * t - directions                            # [speeds, headings] Vector
    return angles, core.residuals(wavevectors[..., None], cases), speeds


def drag() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The speeds of the glass `[drags]` and the speeds of light through it, with the glass and against
    it `[2, drags]`, found from the wave map; and on a fine grid of speeds of the glass, the relativistic
    sum of the two speeds and Fresnel's first-order drag `[2, curve]`."""
    fractions = np.linspace(*drag_range, n_drags)
    speeds = np.linspace(*drag_speed_range, n_drag_speeds)
    glass, _, _ = media()
    wavevectors = speeds[:, None, None] * t - senses[:, None] * z             # [speeds, 2, 1] Vector
    found = core.deepest(speeds, core.residuals(wavevectors, moving(glass, fractions)))   # [2, drags]

    light = expected_speeds()[0, 0]
    curve = np.linspace(*drag_range, n_curve)
    # Fresnel's drag: light gains the fraction 1 - 1 / n**2 of the glass's speed.
    first_order = light + senses[:, None] * curve * (1 - light ** 2)
    return fractions, found, curve, relativistic_sum(curve), first_order


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.electromagnetism.constitutive import render

    glass_modes, crystal_modes = modes()
    speeds, residual, expected = dispersion()
    fractions, found, curve, relativistic, first_order = drag()
    save_figure(render.draw_wave_comparison_figure(glass_modes, crystal_modes), "constitutive_birefringence_3d")
    save_figure(render.draw_dispersion_figure(speeds, residual, expected), "constitutive_dispersion")
    save_figure(render.draw_polarizations_figure(crystal_modes), "constitutive_polarizations")
    save_figure(render.draw_fresnel_surface_figure(*headings(), threshold), "constitutive_fresnel_surface")
    save_figure(render.draw_fresnel_drag_figure(fractions, found, curve, relativistic, first_order),
                "constitutive_fresnel_drag")
    save_animation(render.animate_wave_comparison(glass_modes, crystal_modes, n_frames), "constitutive_comparison",
                   frame_duration)

    # checks
    glass, crystal, _ = media()
    # Every medium carries its waves along z at the expected speeds.
    step = speeds[1] - speeds[0]
    for dips, speed in zip(np.moveaxis(core.dips(speeds, residual, threshold), -1, 0), expected):
        np.testing.assert_allclose(speeds[1:-1][dips], np.unique(speed), atol=step)
    # The crystal's waves satisfy both source-free Maxwell equations.
    crystal_speeds, fields = crystal_modes
    wavevectors = crystal_speeds * t - z                                      # [2] Vector
    np.testing.assert_allclose((wavevectors ^ fields).kernel, 0.0, atol=1e-12)
    np.testing.assert_allclose((wavevectors ^ crystal(fields)).kernel, 0.0, atol=1e-12)
    # The axion term changes no wave's speed.
    axion = glass + mv(core.AxionCoupling, [axion_coupling]) * Bivector
    axion_dips = core.dips(speeds, core.residuals(speeds * t - z, axion), threshold)
    np.testing.assert_allclose(speeds[1:-1][axion_dips], np.unique(expected[0]), atol=step)
    # Light in the moving glass travels at the relativistic sum of the two speeds.
    drag_step = (drag_speed_range[1] - drag_speed_range[0]) / (n_drag_speeds - 1)
    np.testing.assert_allclose(found, relativistic_sum(fractions), atol=drag_step)


if __name__ == "__main__":
    main()
