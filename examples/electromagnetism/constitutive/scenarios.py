"""Glass, a crystal and a ferrite: wave speeds along z and by heading, travelling wave fields, and Fresnel drag in moving glass."""

from __future__ import annotations

import numpy as np

from numga import dim, stack

from examples.electromagnetism.constitutive import core

mv, t, t_s, z, Field = core.mv, core.t, core.t_s, core.z, core.Field
c, mu_0, eps_0, wavenumber = core.c, core.mu_0, core.eps_0, core.wavenumber
Modes = tuple[core.Scalar, core.Field]

# Rest-frame permittivities along x, y, z and inverse permeabilities across yz, zx, xy for glass, a crystal, and a ferrite.
permittivities = np.array([[2.25, 2.25, 2.25], [2.25, 1.5, 1.5], [2.25, 2.25, 2.25]]) * eps_0
inverse_permeabilities = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 0.5, 1.0]]) / mu_0
n_media = len(permittivities)
# Moving-glass speed.
glass_speed = 0.35 * c
# Trial wave speeds along z (as fractions of c), and the residual threshold under which a local minimum counts as a wave.
speed_range = (0.05, 1.5)
n_speeds = 3001
threshold = 4e-3 * wavenumber
# Headings through the xz plane, and trial wave speeds along each heading (as fractions of c).
n_headings = 72
heading_speed_range = (0.4, 1.0)
n_heading_speeds = 601
# Propagation senses with the glass and against it.
senses = np.array([1.0, -1.0])
n_senses = len(senses)
# Glass speed fractions for the drag scan, trial light speeds through it, and analytic curve samples.
drag_range = (-0.6, 0.6)
n_drags = 15
drag_speed_range = (0.05, 1.2)
n_drag_speeds = 1151
n_curve = 200
# Propagation samples along z, frames across one wave period, and frame duration in milliseconds.
n_samples = 200
z_max = mv.scalar(4.0 * np.pi, unit=dim.length)
distances = np.linspace(0.0, 1.0, n_samples) * z_max
n_frames = 30
frame_duration_ms = 40


# --- math -----------------------------------------------------------------------------
# Glass, the crystal, and the ferrite at rest.
media = core.medium(permittivities, inverse_permeabilities)                     # [n_media] Antibivector- <- Bivector+, 1/permeability
# Along z, an electric field along x feels permittivity along x and magnetic plane zx feels inverse permeability across y;
# an electric field along y feels the complementary pair.
expected_speeds = (inverse_permeabilities[:, [1, 0]] / permittivities[:, [0, 1]]).square_root()  # [n_media, 2] Scalar+, length/time


def relativistic_sum(velocities: core.Scalar) -> core.Scalar:
    """Relativistic velocity addition combines the rest-frame phase speed with `velocities` (`[2, ...] Scalar+`, `length/time`) for co- and counter-propagating waves."""
    light = expected_speeds[0, 0]
    flow = senses[:, None] * velocities                                         # [n_senses, ...] Scalar+, length/time
    return (light + flow) / (1 + light * flow / c**2)


def dispersion() -> tuple[core.Scalar, core.Scalar, core.Scalar]:
    """Wave residuals (`[n_speeds, n_media] Scalar+`, `1/length`) across trial speeds (`[n_speeds] Scalar+`) vanish at the analytic z-axis speeds (`[n_media, 2] Scalar+`)."""
    speeds = np.linspace(*speed_range, n_speeds) * c
    wavevectors = wavenumber * (speeds * t_s - z)                               # [n_speeds] Vector+, 1/length
    return speeds, core.wave_residuals(wavevectors[:, None], media), expected_speeds


def propagation_modes() -> tuple[Modes, Modes]:
    """Trailing singular vectors at the z-axis phase speeds give the two eigenmode fields (`[2] Bivector+`) in isotropic glass and the crystal."""
    glass, crystal, _ = media
    glass_speeds, crystal_speeds, _ = expected_speeds
    _, _, glass_fields = core.wave_map(wavenumber * (glass_speeds[0] * t_s - z), glass).svd()
    return (glass_speeds, glass_fields[-2:]), (crystal_speeds, core.field_modes(wavenumber * (crystal_speeds * t_s - z), crystal))


def heading_dispersion() -> tuple[np.ndarray, core.Scalar, core.Scalar]:
    """Wave residuals (`[n_heading_speeds, n_headings, 3] Scalar+`, `1/length`) vary with heading angle (`[n_headings]`) across the xz plane for glass, crystal, and moving glass."""
    angles = np.linspace(0, 2 * np.pi, n_headings)
    speeds = np.linspace(*heading_speed_range, n_heading_speeds) * c
    # Spatial wavevector covectors turned from z toward x across the xz plane.
    slices = (mv.zx * (-angles / 2)).exp() >> z                                 # [n_headings] Vector+
    glass, crystal, _ = media
    cases = stack([glass, crystal, core.moving_medium(glass, glass_speed)])     # [3] Antibivector- <- Bivector+, 1/permeability
    wavevectors = wavenumber * (speeds[:, None] * t_s - slices)                 # [n_heading_speeds, n_headings] Vector+, 1/length
    return angles, core.wave_residuals(wavevectors[..., None], cases), speeds


def fresnel_drag() -> tuple[core.Scalar, core.Scalar, core.Scalar, core.Scalar, core.Scalar]:
    """Wave-map minima across moving-glass velocities (`[n_drags] Scalar+`) track the exact relativistic phase speeds (`[2, n_curve] Scalar+`) beyond first-order Fresnel drag."""
    velocities = np.linspace(*drag_range, n_drags) * c
    speeds = np.linspace(*drag_speed_range, n_drag_speeds) * c
    glass, _, _ = media
    wavevectors = wavenumber * (speeds[:, None, None] * t_s - senses[:, None] * z)  # [n_drag_speeds, n_senses, 1] Vector+, 1/length
    found_speeds = core.minimum_speeds(speeds, core.wave_residuals(wavevectors, core.moving_medium(glass, velocities)))  # [n_senses, n_drags] Scalar+, length/time

    light = expected_speeds[0, 0]
    curve = np.linspace(*drag_range, n_curve) * c
    # In Fresnel's first-order drag, light picks up the fraction 1 - 1 / n**2 of the medium's speed.
    first_order = light + senses[:, None] * curve * (1 - (light / c) ** 2)
    return velocities, found_speeds, curve, relativistic_sum(curve), first_order


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.electromagnetism.constitutive import render
    from examples.util.animation import save_animation, save_figure

    glass_modes, crystal_modes = propagation_modes()
    speeds, residual, expected = dispersion()
    velocities, found_speeds, curve, relativistic, first_order = fresnel_drag()
    save_figure(render.wave_comparison_figure(glass_modes, crystal_modes, distances), "constitutive_birefringence_3d")
    save_figure(render.dispersion_figure(speeds, residual, expected), "constitutive_dispersion")
    save_figure(render.polarizations_figure(crystal_modes), "constitutive_polarizations")
    save_figure(render.fresnel_surface_figure(*heading_dispersion(), threshold), "constitutive_fresnel_surface")
    save_figure(render.fresnel_drag_figure(velocities, found_speeds, curve, relativistic, first_order),
                "constitutive_fresnel_drag")

    glass_stream = core.travelling_fields(*glass_modes, distances, n_frames)
    crystal_stream = core.travelling_fields(*crystal_modes, distances, n_frames)
    streamed_pairs = []

    def recording():
        for pair in zip(glass_stream, crystal_stream):
            streamed_pairs.append(pair)
            yield pair

    save_animation(
        render.wave_comparison_frames(glass_modes, crystal_modes, distances, recording()),
        "constitutive_comparison",
        frame_duration_ms,
    )

    # checks
    _, crystal, _ = media
    # Every medium carries its waves along z at the expected speeds.
    speed_step = speeds[1] - speeds[0]
    for medium_minima, speed in zip(np.moveaxis(core.local_minima(residual, threshold), -1, 0), expected):
        found = speeds[1:-1][medium_minima]
        targets = stack([speed[0]]) if (speed[0] - speed[1]).abs() < 1e-12 * c else speed
        assert ((found - targets).abs() < speed_step).all()
    # The crystal's waves satisfy both source-free Maxwell equations.
    crystal_speeds, fields = crystal_modes
    wavevectors = wavenumber * (crystal_speeds * t_s - z)                       # [2] Vector+, 1/length
    for residual_term in (wavevectors ^ fields, mu_0 * (wavevectors ^ crystal(fields))):
        assert (residual_term.reverse().scalar_product(t * residual_term.involute() * t.inverse()) < 1e-24 * wavenumber**2).all()
    # Streamed travelling fields stay finite and bounded across the period.
    assert len(streamed_pairs) == n_frames
    first_field = streamed_pairs[0][1]
    assert (first_field.reverse().scalar_product(t * first_field.involute() * t.inverse()) > 0).any()
    # Light in moving glass travels at the relativistic sum of the two speeds.
    drag_step = (drag_speed_range[1] - drag_speed_range[0]) / (n_drag_speeds - 1) * c
    assert ((found_speeds - relativistic_sum(velocities)).abs() < drag_step).all()


if __name__ == "__main__":
    main()
